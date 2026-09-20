/* Include from rtkpos.c AFTER RTKLIB state-index macros. Requires a rebuilt rtk_t
 * containing void *learned_qr. The unchanged DD covariance builder consumes the
 * resulting SD variances. No modification of integer ambiguity states or search.
 */
#ifndef RTKLIB_LEARNED_QR_ADAPTER_H
#define RTKLIB_LEARNED_QR_ADAPTER_H
#include "learned_qr.h"
#include "qr_trace.h"

typedef struct {
    qr_model model;
    qr_history q;
    qr_history *r;
    int mode, loaded, stable;
    qr_callback callback;
    void *user;
    int failed;
    int observe_candidates, observe_fixed;
    double innovation[MAXSAT][NFREQ];
    struct { double t,mp0,p,d; unsigned signal; int valid,mpvalid; } evidence[MAXSAT][NFREQ];
    FILE *log;
} qr_rtk_context;

static inline int qr_notify(qr_rtk_context *c, const qr_event *e)
{
    if (!c || !c->callback || c->failed) return 0;
    if (c->callback(c->user,e)) {c->failed=1; return -1;}
    return 0;
}
static inline void qr_emit(rtk_t *rtk,int kind,int n,int m,const int *ix,
                           const double *a,double *b,const double *other)
{
    qr_rtk_context *c=(qr_rtk_context*)rtk->learned_qr;
    qr_event e;
    int week;
    if (!c || !c->callback || c->failed) return;
    memset(&e,0,sizeof(e));e.kind=kind;e.nx=rtk->nx;e.n=n;e.m=m;
    e.time=time2gpst(rtk->sol.time,&week);e.time+=week*604800.0;
    e.ix=ix;e.x=rtk->x;e.P=rtk->P;e.a=a;e.b=b;e.c=other;
    qr_notify(c,&e);
}
static inline void qr_filter_event(rtk_t *rtk,double *x,double *P,
                                   const double *H,double *v,const double *R,
                                   const int *flags,int nv)
{
    qr_rtk_context *c=(qr_rtk_context*)rtk->learned_qr;
    qr_event e;
    if (!c || !c->callback || c->failed) return;
    memset(&e,0,sizeof(e));e.kind=QR_FILTER;e.nx=rtk->nx;e.n=nv;
    e.x=x;e.P=P;e.a=H;e.b=v;e.c=R;e.ix=flags;
    qr_notify(c,&e);
}
static inline void qr_sd_event(qr_rtk_context *c,int sat,int channel,
                               double variance,double factor)
{
    qr_event e;int ix[2]={sat,channel};
    if (!c || !c->callback || c->failed) return;
    memset(&e,0,sizeof(e));e.kind=QR_SD_VARIANCE;e.n=1;e.ix=ix;
    e.a=&variance;e.b=&factor;qr_notify(c,&e);
}

static inline void *qr_rtk_create(void)
{
    const char *mode=getenv("RTKLIB_QR_MODE"),*path=getenv("RTKLIB_QR_MODEL");
    const char *logpath=getenv("RTKLIB_QR_LOG"),*test=getenv("RTKLIB_QR_ALLOW_TEST_MODEL");
    qr_rtk_context *c;
    int requested=0,j;
    const char *stable=getenv("RTKLIB_QR_STABLE");
    if(mode && strcmp(mode,"off")!=0) {
        if(strcmp(mode,"q")==0) requested=1;
        else if(strcmp(mode,"r")==0) requested=2;
        else if(strcmp(mode,"qr")==0) requested=3;
        else {fprintf(stderr,"learned-qr: invalid mode; using native Q/R\n");return NULL;}
    }
    if(!requested && (!logpath || !*logpath) && (!stable || strcmp(stable,"1")!=0)) return NULL;
    c=(qr_rtk_context*)calloc(1,sizeof(*c));
    if(!c) return NULL;
    c->stable=requested || (stable && strcmp(stable,"1")==0);
    c->r=(qr_history*)calloc(MAXSAT*2*NFREQ,sizeof(qr_history));
    if(!c->r) {free(c);return NULL;}
    if(requested) {
        c->loaded=qr_load(&c->model,path,test && strcmp(test,"1")==0);
        if(c->loaded) c->mode=requested;
        else fprintf(stderr,"learned-qr: no valid permitted model; using native Q/R\n");
    }
    if(!c->loaded) {c->model.max_gap=30;c->stable=stable && strcmp(stable,"1")==0;}
    if(logpath && *logpath) {
        c->log=fopen(logpath,"w");
        if(c->log) {
            fprintf(c->log,"kind,t_gpst_s,sat,channel,scale0,scale1");
            for(j=0;j<QR_RDIM;j++) fprintf(c->log,",f%d",j);
            fprintf(c->log,"\n");
        } else fprintf(stderr,"learned-qr: cannot open feature log\n");
    }
    return c;
}
static inline void qr_rtk_destroy(void *context)
{
    qr_rtk_context *c=(qr_rtk_context*)context;
    if(!c) return;
    if(c->log) fclose(c->log);
    free(c->r);free(c);
}
static inline void qr_rtk_reset(void *context)
{
    qr_rtk_context *c=(qr_rtk_context*)context;
    if(!c) return;
    memset(c->evidence,0,sizeof(c->evidence));
    memset(c->innovation,0,sizeof(c->innovation));
    memset(&c->q,0,sizeof(c->q));
    memset(c->r,0,MAXSAT*2*NFREQ*sizeof(qr_history));
}
static inline double qr_seconds(gtime_t time)
{
    int week;
    double tow=time2gpst(time,&week);
    return week*604800.0+tow;
}
static inline int qr_sys_index(int sys)
{
    switch(sys) {
        case SYS_GPS:return 0;case SYS_GLO:return 1;case SYS_GAL:return 2;
        case SYS_CMP:return 3;case SYS_QZS:return 4;case SYS_SBS:return 5;
        case SYS_IRN:return 6;default:return -1;
    }
}
static inline void qr_capture(qr_rtk_context *c,qr_history *h,int isq,double t,
                              unsigned signature,int reset,const double *features,
                              int sat,int channel,double *factors)
{
    double flat[QR_INPUT_MAX];
    int dim=isq?QR_QDIM:QR_RDIM,j,changed;
    factors[0]=factors[1]=1;
    changed=qr_history_push(h,t,signature,reset,features,dim,c->model.max_gap);
    if(changed<0) return;
    if(!h->cached_valid) {
        qr_history_flat(h,dim,flat);
        h->cached[0]=h->cached[1]=1;
        if(c->callback && !c->loaded && (c->mode & (isq?1:2))) {
            qr_event event;int ix[2]={sat,channel};
            memset(&event,0,sizeof(event));
            event.kind=isq?QR_INFER_Q:QR_INFER_R;event.n=dim;event.m=isq?2:1;
            event.time=t;event.a=flat;event.b=h->cached;event.ix=ix;
            qr_notify(c,&event);
            for(j=0;j<event.m;j++) {
                double cap=isq?100.0:1e6;
                if(!isfinite(h->cached[j]) || h->cached[j]<1.0/cap || h->cached[j]>cap)
                    c->failed=1;
            }
        }
        else if(c->loaded && (c->mode & (isq?1:2)))
            qr_model_forward(&c->model,isq,flat,h->cached);
        h->cached_valid=1;
    }
    factors[0]=h->cached[0];factors[1]=h->cached[1];
    if(changed>0 && c->log) {
        fprintf(c->log,"%c,%.9f,%d,%d,%.17g,%.17g",isq?'Q':'R',t,sat,channel,factors[0],factors[1]);
        for(j=0;j<QR_RDIM;j++) fprintf(c->log,",%.17g",j<dim?features[j]:0.0);
        fprintf(c->log,"\n");
    }
}

/* Input Q is the existing LOCAL ENU acceleration driving covariance (3x3).
 * Preserve the native placement, time discretization, and ENU->ECEF rotation.
 * Only existing horizontal/vertical variances are scaled, not all P entries. */
static inline void qr_apply_q(rtk_t *rtk,double dt,double *Q)
{
    qr_rtk_context *c=(qr_rtk_context*)rtk->learned_qr;
    double pos[3],vel[3],acc[3],f[QR_QDIM],s[2],variance;
    int j;
    if(!c || !rtk->opt.dynamics || rtk->nx<9 || dt<=0 || !isfinite(dt) ||
       rtk->opt.mode<=PMODE_DGPS || rtk->opt.mode>PMODE_FIXED) return;
    ecef2pos(rtk->x,pos);ecef2enu(pos,rtk->x+3,vel);ecef2enu(pos,rtk->x+6,acc);
    f[0]=log1p(dt);
    for(j=0;j<3;j++) {f[j+1]=vel[j]/30.0;f[j+4]=acc[j]/3.0;}
    variance=(rtk->P[0]+rtk->P[1+rtk->nx]+rtk->P[2+2*rtk->nx])/3.0;
    f[7]=log1p(sqrt(fmax(variance,0)));f[8]=rtk->sol.ns/40.0;f[9]=rtk->sol.stat/6.0;
    qr_capture(c,&c->q,1,qr_seconds(rtk->sol.time),0,0,f,0,0,s);
    if(c->mode&1) {Q[0]*=s[0];Q[4]*=s[0];Q[8]*=s[1];}
}

/* Current SD code innovations: remove a per-system clock with a median,
 * normalize by predicted geometry and ORIGINAL code noise, never a learned R.
 * No reference trajectory or future observation is consumed here. */
static inline void qr_prepare_evidence(rtk_t *rtk,const int *sat,const double *y,
    const double *e,const double *azel,const int *iu,const int *ir,int ns,int nf,const double *P)
{
    qr_rtk_context *c=(qr_rtk_context*)rtk->learned_qr;
    double sorted[MAXSAT],med,var,vp,a,b,v; int i,j,f,sys,n,q;
    if(!c || c->model.rpolicy<2) return;
    memset(c->innovation,0,sizeof(c->innovation));
    for(sys=0;sys<7;sys++) for(f=0;f<nf;f++) {
        n=0;
        for(i=0;i<ns;i++) if(qr_sys_index(rtk->ssat[sat[i]-1].sys)==sys &&
            y[f+nf+iu[i]*nf*2]!=0 && y[f+nf+ir[i]*nf*2]!=0) {
            sorted[n++]=y[f+nf+iu[i]*nf*2]-y[f+nf+ir[i]*nf*2];
        }
        if(n<4) continue;
        for(i=1;i<n;i++) {v=sorted[i];j=i;while(j>0 && sorted[j-1]>v){sorted[j]=sorted[j-1];j--;}sorted[j]=v;}
        med=n%2?sorted[n/2]:.5*(sorted[n/2-1]+sorted[n/2]);
        for(i=0;i<ns;i++) if(qr_sys_index(rtk->ssat[sat[i]-1].sys)==sys &&
            y[f+nf+iu[i]*nf*2]!=0 && y[f+nf+ir[i]*nf*2]!=0) {
            a=rtk->opt.eratio[f]*rtk->opt.err[1];b=rtk->opt.eratio[f]*rtk->opt.err[2];
            var=2*(a*a+b*b/fmax(.04,pow(sin(azel[1+iu[i]*2]),2)));vp=0;
            if(P) for(j=0;j<3;j++)for(q=0;q<3;q++)vp+=e[j+iu[i]*3]*P[j+q*rtk->nx]*e[q+iu[i]*3];
            c->innovation[sat[i]-1][f]=(y[f+nf+iu[i]*nf*2]-y[f+nf+ir[i]*nf*2]-med)/sqrt(fmax(1.,var+vp));
        }
    }
}

/* SD code/phase variances are already in m^2 in varerr(). Do not multiply by
 * wavelength or square these factors again. The original half-cycle additions
 * remain outside this hook, just as they are in upstream ddres(). */
static inline double qr_apply_r(rtk_t *rtk,int sat,int sys,double el,
                                double snr_rover,double snr_base,double bl,double age,
                                int f,const prcopt_t *opt,const obsd_t *obs,
                                const obsd_t *baseobs,double native_variance)
{
    qr_rtk_context *c=(qr_rtk_context*)rtk->learned_qr;
    qr_history *h;
    double feature[QR_RDIM],factor[2],t,cn0,delta=0;
    unsigned signature;
    int nf=opt->nf,frq,code,index,slip,sysindex=qr_sys_index(sys),continuous;
    if(!c || !obs || !baseobs || sat<1 || sat>MAXSAT || nf<1 || nf>NFREQ ||
       f<0 || f>=2*nf || opt->ionoopt==IONOOPT_IFLC || rtk->tt<0 ||
       opt->mode<=PMODE_DGPS || opt->mode>PMODE_FIXED || sysindex<0) return native_variance;
    frq=f%nf;code=f>=nf;index=(sat-1)*2*NFREQ+code*NFREQ+frq;
    h=&c->r[index];t=qr_seconds(obs->time);
    signature=((unsigned)obs->code[frq]<<8)|(unsigned)baseobs->code[frq];
    if(h->count && t==h->last_time && signature==h->signature && h->cached_valid) {
        qr_sd_event(c,sat,code*NFREQ+frq,native_variance,(c->mode&2)?h->cached[0]:1.0);
        return (c->mode&2)?native_variance*h->cached[0]:native_variance;
    }
    slip=((obs->LLI[frq]|baseobs->LLI[frq]|rtk->ssat[sat-1].slip[frq])&LLI_SLIP)!=0;
    continuous=qr_history_continuous(h,t,signature,slip,c->model.max_gap);
    cn0=(snr_rover-40)/10.0;
    if(continuous) delta=cn0-h->values[(h->next+QR_WINDOW-1)%QR_WINDOW][0];
    feature[0]=cn0;feature[1]=(snr_base-40)/10.0;feature[2]=sin(el);
    feature[3]=(double)code;feature[4]=(double)frq/fmax(NFREQ-1,1);feature[5]=sysindex/6.0;
    feature[6]=log1p(fabs(bl)/1000);feature[7]=qr_signed_log(age);
    feature[8]=qr_signed_log(code?obs->Pstd[frq]:obs->Lstd[frq]);
    feature[9]=((obs->LLI[frq]|baseobs->LLI[frq])&LLI_HALFC)?1:0;
    feature[10]=delta;feature[11]=log1p(continuous?h->seen+1:1);
    if(c->model.rpolicy>=2) {
        double lam;
        double f1=FREQL1,f2=FREQL2,gamma=(f1/f2)*(f1/f2),mp,dt,psd,dsd,anom=0.,pd=0.;
        unsigned mp_signal=((unsigned)obs->code[0]<<24)|((unsigned)obs->code[1]<<16)|((unsigned)baseobs->code[0]<<8)|baseobs->code[1];
        int mp_slip=((obs->LLI[0]|obs->LLI[1]|baseobs->LLI[0]|baseobs->LLI[1]|rtk->ssat[sat-1].slip[0]|rtk->ssat[sat-1].slip[1])&LLI_SLIP)!=0;
        int valid=(sys==SYS_GPS && obs->P[frq]!=0 && baseobs->P[frq]!=0 &&
            obs->L[0]!=0 && obs->L[1]!=0 && baseobs->L[0]!=0 && baseobs->L[1]!=0 && frq<2);
        lam=CLIGHT/(frq==0?f1:f2);
        feature[8]=qr_signed_log(c->innovation[sat-1][frq]);feature[9]=feature[10]=0.;
        if(valid) {
            psd=obs->P[frq]-baseobs->P[frq];dsd=obs->D[frq]-baseobs->D[frq];
            mp=psd-lam*(obs->L[frq]-baseobs->L[frq])-
              2.*(frq==0?1.:gamma)/(gamma-1.)*(CLIGHT/f1*(obs->L[0]-baseobs->L[0])-CLIGHT/f2*(obs->L[1]-baseobs->L[1]));
            dt=t-c->evidence[sat-1][frq].t;
            if(!c->evidence[sat-1][frq].valid || dt>c->model.max_gap || dt<0 ||
              c->evidence[sat-1][frq].signal!=mp_signal || (mp_slip && dt>0)) {
                c->evidence[sat-1][frq].mp0=mp;c->evidence[sat-1][frq].valid=1;
            }
            else {
                anom=(mp-c->evidence[sat-1][frq].mp0)/sqrt(fmax(1.,code?native_variance:1.));
                if(dt>0 && obs->D[frq]!=0 && baseobs->D[frq]!=0)
                    pd=(psd-c->evidence[sat-1][frq].p+.5*lam*(dsd+c->evidence[sat-1][frq].d)*dt)/sqrt(fmax(1.,code?native_variance:1.));
            }
            /* Only code consumes/updates the anchor. Carrier pass must not overwrite it. */
            if(code) {
                if(dt>0 && dt<=c->model.max_gap) c->evidence[sat-1][frq].mp0+=fmin(1.,dt/600.)*fmax(-1.,fmin(1.,mp-c->evidence[sat-1][frq].mp0));
                c->evidence[sat-1][frq].t=t;c->evidence[sat-1][frq].p=psd;c->evidence[sat-1][frq].d=dsd;
                c->evidence[sat-1][frq].signal=mp_signal;
            }
            feature[9]=qr_signed_log(anom);feature[10]=qr_signed_log(pd);
        } else if(code) c->evidence[sat-1][frq].valid=0;
    }
    qr_capture(c,h,0,t,signature,slip,feature,sat,code*NFREQ+frq,factor);
    qr_sd_event(c,sat,code*NFREQ+frq,native_variance,(c->mode&2)?factor[0]:1.0);
    return (c->mode&2)?native_variance*factor[0]:native_variance;
}
#endif
