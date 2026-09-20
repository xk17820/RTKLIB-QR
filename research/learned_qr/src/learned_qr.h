/* Learned Q/R numerical runtime, feature/model schema V1.
 * Header-only C99. No global mutable state, external ML dependency or bias head.
 * New code supplied under the MIT license in LICENSE-QR; upstream unchanged.
 */
#ifndef RTKLIB_LEARNED_QR_H
#define RTKLIB_LEARNED_QR_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QR_WINDOW 10
#define QR_QDIM 10
#define QR_RDIM 12
#define QR_HIDDEN 16
#define QR_INPUT_MAX (QR_WINDOW*(QR_RDIM+1))

typedef struct {
    int input, output;
    double w1[QR_HIDDEN][QR_INPUT_MAX], b1[QR_HIDDEN];
    double w2[2][QR_HIDDEN], b2[2];
} qr_net;

typedef struct {
    qr_net q, r;
    double qlimit, rlimit, max_gap;
    int rpolicy; /* V1=0 legacy, V2=1 CN0 guard, V3=2 consistency/3 blind/4 signed-code ablation */
    double guard_anchor,guard_width,phase_cap;
    int provenance; /* 0=untrained, 1=caller-declared real trained, 2=synthetic */
} qr_model;

typedef struct {
    double values[QR_WINDOW][QR_RDIM];
    double last_time, cached[2];
    unsigned signature;
    int count, next, seen, cached_valid;
} qr_history;

static inline int qr_read_double(FILE *f,double *x)
{
    return fscanf(f,"%lf",x)==1 && isfinite(*x) && fabs(*x)<=1e6;
}

static inline int qr_read_net(FILE *f,qr_net *net,int dim,int outputs)
{
    int i,j;
    if (fscanf(f,"%d %d",&net->input,&net->output)!=2 ||
        net->input!=QR_WINDOW*(dim+1) || net->output!=outputs) return 0;
    for (i=0;i<QR_HIDDEN;i++) for (j=0;j<net->input;j++)
        if (!qr_read_double(f,&net->w1[i][j])) return 0;
    for (i=0;i<QR_HIDDEN;i++) if (!qr_read_double(f,&net->b1[i])) return 0;
    for (i=0;i<outputs;i++) for (j=0;j<QR_HIDDEN;j++)
        if (!qr_read_double(f,&net->w2[i][j])) return 0;
    for (i=0;i<outputs;i++) if (!qr_read_double(f,&net->b2[i])) return 0;
    return 1;
}

static inline int qr_load(qr_model *m,const char *path,int allow_test)
{
    FILE *f;
    char magic[40],extra;
    int ok;
    if (!m || !path || !(f=fopen(path,"r"))) return 0;
    memset(m,0,sizeof(*m));
    ok=fscanf(f,"%39s %d %lf %lf %lf",magic,&m->provenance,
              &m->qlimit,&m->rlimit,&m->max_gap)==5;
    ok=ok && (strcmp(magic,"RTKLIB_QR_V1")==0 || strcmp(magic,"RTKLIB_QR_V2")==0 || strcmp(magic,"RTKLIB_QR_V3")==0) &&
       m->provenance>=0 && m->provenance<=2 &&
       (m->provenance==1 || allow_test) &&
       isfinite(m->qlimit) && m->qlimit>1 && m->qlimit<=100 &&
       isfinite(m->rlimit) && m->rlimit>1 && m->rlimit<=1e6 &&
       isfinite(m->max_gap) && m->max_gap>0 && m->max_gap<=3600;
    if (ok && (strcmp(magic,"RTKLIB_QR_V2")==0 || strcmp(magic,"RTKLIB_QR_V3")==0)) {
        ok=fscanf(f,"%d %lf %lf %lf",&m->rpolicy,&m->guard_anchor,&m->guard_width,&m->phase_cap)==4;
        ok=ok && ((strcmp(magic,"RTKLIB_QR_V2")==0 && m->rpolicy==1) ||
                  (strcmp(magic,"RTKLIB_QR_V3")==0 && (m->rpolicy>=2 && m->rpolicy<=4))) && isfinite(m->guard_anchor) &&
            (m->rpolicy==1?m->guard_anchor>=20:m->guard_anchor>0) && m->guard_anchor<=55 && isfinite(m->guard_width) &&
            m->guard_width>=1 && m->guard_width<=30 && isfinite(m->phase_cap) &&
            m->phase_cap>=1 && m->phase_cap<=100 && (m->rpolicy<2 || m->phase_cap==1.);
    }
    if (ok) ok=qr_read_net(f,&m->q,QR_QDIM,2) && qr_read_net(f,&m->r,QR_RDIM,1);
    if (ok && fscanf(f," %c",&extra)==1) ok=0; /* reject extra/corrupt content */
    fclose(f);
    if (!ok) memset(m,0,sizeof(*m));
    return ok;
}

static inline int qr_forward(const qr_net *net,const double *x,double limit,double *out)
{
    int i,j;
    double hidden[QR_HIDDEN],z;
    out[0]=1.0;
    if (net->output==2) out[1]=1.0;
    if (net->input<=0 || net->input>QR_INPUT_MAX || net->output<1 || net->output>2 ||
        !isfinite(limit) || limit<=1) return 0;
    for (j=0;j<net->input;j++) if (!isfinite(x[j]) || fabs(x[j])>1e6) return 0;
    for (i=0;i<QR_HIDDEN;i++) {
        z=net->b1[i];
        for (j=0;j<net->input;j++) z+=net->w1[i][j]*x[j];
        if (!isfinite(z)) return 0;
        hidden[i]=tanh(z);
    }
    for (i=0;i<net->output;i++) {
        z=net->b2[i];
        for (j=0;j<QR_HIDDEN;j++) z+=net->w2[i][j]*hidden[j];
        if (!isfinite(z)) {out[0]=1; if(net->output==2) out[1]=1; return 0;}
        out[i]=exp(log(limit)*tanh(z));
    }
    return 1;
}

/* Versioned policy: exactly the same transform as the Torch model. */
static inline int qr_model_forward(const qr_model *m,int isq,const double *x,double *out)
{
    int i,j,k=(QR_WINDOW-1)*(QR_RDIM+1);
    double gate=0.0,cn0,d,xx[QR_INPUT_MAX];const double *input=x;
    if(!isq && m->rpolicy==3){memcpy(xx,x,sizeof(double)*m->r.input);
        for(i=0;i<QR_WINDOW;i++) { for(j=8;j<=10;j++)xx[i*(QR_RDIM+1)+j]=0.; }
        input=xx;
    }
    int ok=qr_forward(isq?&m->q:&m->r,input,isq?m->qlimit:m->rlimit,out);
    if (!ok || isq || m->rpolicy==0) return ok;
    if (x[k+QR_RDIM]<=0.5) {out[0]=1.;return 1;}
    if(m->rpolicy>=2) {
        if(x[k+3]<.5){out[0]=1.;return 1;}
        d=fmax(expm1(fmin(8.,fabs(x[k+8]))),expm1(fmin(8.,fabs(x[k+9]))));
        gate=m->rpolicy==3?1.:fmin(1.,fmax(0.,(d-m->guard_anchor)/m->guard_width));
        out[0]=1.+gate*(m->rpolicy==4?out[0]-1.:fmax(out[0]-1.,0.));return 1;
    }
    for(i=0;i<2;i++) {
        cn0=40.+10.*x[k+i];
        d=cn0>0.?(m->guard_anchor-cn0)/m->guard_width:0.;
        gate=fmax(gate,d);
    }
    gate=fmin(1.,fmax(0.,gate));
    out[0]=1.+gate*(x[k+3]>.5?fmax(out[0]-1.,0.):m->phase_cap-1.);
    return 1;
}

static inline int qr_history_continuous(const qr_history *h,double t,unsigned signature,
                                        int reset,double max_gap)
{
    return h->count && !reset && signature==h->signature &&
           t>h->last_time && t-h->last_time<=max_gap;
}

/* Returns 1=new step, 0=duplicate epoch, -1=invalid input. Repeated ddres calls
 * in the same epoch MUST NOT advance history or repeatedly consume a slip flag. */
static inline int qr_history_push(qr_history *h,double t,unsigned signature,
                                 int reset,const double *x,int dim,double max_gap)
{
    int i;
    if (!h || !x || dim<1 || dim>QR_RDIM || !isfinite(t) ||
        !isfinite(max_gap) || max_gap<=0) return -1;
    for(i=0;i<dim;i++) if(!isfinite(x[i]) || fabs(x[i])>1e6) return -1;
    if (h->count && t==h->last_time && signature==h->signature) return 0;
    if (!qr_history_continuous(h,t,signature,reset,max_gap)) memset(h,0,sizeof(*h));
    for(i=0;i<dim;i++) h->values[h->next][i]=x[i];
    h->next=(h->next+1)%QR_WINDOW;
    if(h->count<QR_WINDOW) h->count++;
    if(h->seen<1000000000) h->seen++;
    h->last_time=t; h->signature=signature; h->cached_valid=0;
    return 1;
}

/* Oldest -> newest, zero padding at the beginning, validity mask per step. */
static inline void qr_history_flat(const qr_history *h,int dim,double *out)
{
    int i,j,index,padding=QR_WINDOW-h->count;
    memset(out,0,(size_t)QR_WINDOW*(dim+1)*sizeof(double));
    for(i=0;i<h->count;i++) {
        index=(h->next-h->count+i+QR_WINDOW)%QR_WINDOW;
        for(j=0;j<dim;j++) out[(padding+i)*(dim+1)+j]=h->values[index][j];
        out[(padding+i)*(dim+1)+dim]=1;
    }
}

static inline double qr_signed_log(double x)
{
    return x<0 ? -log1p(-x) : log1p(x);
}
#endif
