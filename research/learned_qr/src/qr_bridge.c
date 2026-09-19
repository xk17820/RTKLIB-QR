/* RINEX + broadcast-navigation sessions for live FLOAT training and RTK replay.
 * The native rtkpos() runs on CURRENT learned states, not on archived residuals.
 * The callback is optional; without it all computation stays in C.
 * Option parsing in RTKLIB is global: callers serialize opens (Python does).
 * Sessions own observations, navigation, filter state and feature buffers.
 */
#include "qr_bridge.h"
#include "learned_qr_adapter.h"

typedef struct {
    obs_t rover,base;
    nav_t nav;
    prcopt_t opt;
    rtk_t rtk;
    qr_model model;
    int mode,loaded,training,initialized,failed,first,stable;
    int *ri,*bi,nr,nb,cursor;
    double max_gap,pair_age;
    gtime_t previous;
} qr_session;

static int qr_error(char *error,int size,const char *message)
{
    if(error && size>0) snprintf(error,(size_t)size,"%s",message);
    return -1;
}
static int qr_epoch_indices(const obs_t *obs,int **indices)
{
    int i,n=0;
    *indices=(int*)malloc((size_t)(obs->n+1)*sizeof(int));
    if(!*indices) return -1;
    for(i=0;i<obs->n;i++) {
        if(i==0 || fabs(timediff(obs->data[i].time,obs->data[(*indices)[n-1]].time))>DTTOL)
            (*indices)[n++]=i;
    }
    (*indices)[n]=obs->n;
    return n;
}
static int qr_new_filter(qr_session *s)
{
    qr_rtk_context *c;
    if(s->initialized) rtkfree(&s->rtk);
    rtkinit(&s->rtk,&s->opt);s->initialized=1;
    qr_rtk_destroy(s->rtk.learned_qr);s->rtk.learned_qr=NULL;
    c=(qr_rtk_context*)calloc(1,sizeof(*c));
    if(!c) return 0;
    c->r=(qr_history*)calloc(MAXSAT*2*NFREQ,sizeof(qr_history));
    if(!c->r) {free(c);return 0;}
    c->mode=s->mode;c->loaded=s->loaded;c->model=s->model;c->stable=s->stable;
    c->model.max_gap=s->max_gap;s->rtk.learned_qr=c;
    s->first=1;
    return 1;
}
EXPORT int qr_bridge_abi(void) {return 1;}
EXPORT int qr_set_stable(void *session,int enabled)
{
    qr_session *s=(qr_session*)session;
    if(!s || (!enabled && (s->training || s->mode))) return -1;
    s->stable=enabled!=0;
    if(s->initialized && s->rtk.learned_qr)
        ((qr_rtk_context*)s->rtk.learned_qr)->stable=s->stable;
    return 0;
}
EXPORT void qr_close(void *session)
{
    qr_session *s=(qr_session*)session;
    if(!s) return;
    if(s->initialized) rtkfree(&s->rtk);
    freeobs(&s->rover);freeobs(&s->base);freenav(&s->nav,0xFF);
    free(s->ri);free(s->bi);free(s);
}
EXPORT void *qr_open(const char *config,const char *rover,const char *base,
                     const char **navfiles,int nnav,const char *model,int mode,
                     int training,int allow_test,double max_gap,double pair_age,
                     char *error,int error_size)
{
    qr_session *s=NULL;
    sta_t sr={0},sb={0},sn={0};
    solopt_t so;filopt_t fo;
    obs_t dummy={0};
    int i;
    const char *why=NULL;
    if(!config || !rover || !base || !navfiles || nnav<1 || mode<0 || mode>3 ||
       !isfinite(max_gap) || max_gap<=0 || max_gap>3600 ||
       !isfinite(pair_age) || pair_age<0 || pair_age>3600) {
        qr_error(error,error_size,"invalid session arguments");return NULL;
    }
    s=(qr_session*)calloc(1,sizeof(*s));
    if(!s) {qr_error(error,error_size,"out of memory");return NULL;}
    resetsysopts();
    if(!loadopts(config,sysopts)) {why="cannot read RTKLIB config";goto fail;}
    getsysopts(&s->opt,&so,&fo);
    if(s->opt.mode!=PMODE_KINEMA) {why="bridge supports pos1-posmode=kinematic only";goto fail;}
    if(!s->opt.dynamics) {why="bridge requires pos1-dynamics=on";goto fail;}
    if(s->opt.soltype!=0 || s->opt.niter!=1 || s->opt.intpref) {
        why="bridge requires forward, pos2-niter=1 and misc-timeinterp=off";goto fail;
    }
    if(s->opt.ionoopt!=IONOOPT_OFF && s->opt.ionoopt!=IONOOPT_BRDC) {
        why="bridge supports ionoopt off/brdc only, not estimated/IFLC/TEC states";goto fail;
    }
    if(s->opt.tropopt>TROPOPT_SAAS || s->opt.glomodear==GLO_ARMODE_AUTOCAL ||
       s->opt.sateph!=EPHOPT_BRDC || s->opt.tidecorr) {
        why="bridge requires brdc ephemeris, off/saas trop, no GLO autocal and no tides";goto fail;
    }
    if(fo.satantp[0] || fo.rcvantp[0] || fo.stapos[0] || fo.iono[0] || fo.dcb[0] || fo.eop[0]) {
        why="bridge does not load external antenna/station/iono/DCB/ERP files; use the supplied profile";goto fail;
    }
    if(s->opt.baseline[0]>0) {why="bridge does not support a fixed baseline constraint";goto fail;}
    if(s->opt.nf<1 || s->opt.nf>NFREQ || pair_age>s->opt.maxtdiff+1e-9) {
        why="invalid frequency count or pair_age exceeds pos2-maxage";goto fail;
    }
    if(s->opt.refpos!=POSOPT_POS_XYZ && s->opt.refpos!=POSOPT_POS_LLH && s->opt.refpos!=POSOPT_RINEX) {
        why="base position must be explicit LLH/XYZ or rinexhead";goto fail;
    }
    s->stable=1;s->mode=mode;s->training=training;s->max_gap=max_gap;s->pair_age=pair_age;
    if(training) s->opt.modear=ARMODE_OFF; /* deployment config remains unchanged */
    if(model && *model) {
        if(!qr_load(&s->model,model,allow_test)) {why="invalid, untrained or incompatible model";goto fail;}
        if(fabs(s->model.max_gap-max_gap)>1e-9) {why="max_gap must match exported model";goto fail;}
        s->loaded=1;
    }
    if(readrnx(rover,1,s->opt.rnxopt[0],&s->rover,&s->nav,&sr)<=0 || !s->rover.n) {
        why="cannot read rover RINEX observations";goto fail;
    }
    if(readrnx(base,2,s->opt.rnxopt[1],&s->base,&s->nav,&sb)<=0 || !s->base.n) {
        why="cannot read base RINEX observations";goto fail;
    }
    for(i=0;i<nnav;i++) {
        if(!navfiles[i] || readrnx(navfiles[i],1,"",&dummy,&s->nav,&sn)<=0) {
            why="cannot read broadcast navigation RINEX";goto fail;
        }
    }
    freeobs(&dummy);
    if(!s->nav.n && !s->nav.ng) {why="no broadcast ephemerides loaded";goto fail;}
    uniqnav(&s->nav);sortobs(&s->rover);sortobs(&s->base);
    if(s->opt.refpos==POSOPT_RINEX) matcpy(s->opt.rb,sb.pos,3,1);
    if(norm(s->opt.rb,3)<RE_WGS84/2) {why="invalid base ECEF position; set ant2-pos1/2/3";goto fail;}
    s->nr=qr_epoch_indices(&s->rover,&s->ri);s->nb=qr_epoch_indices(&s->base,&s->bi);
    if(s->nr<1 || s->nb<1 || !qr_new_filter(s)) {why="cannot allocate session state";goto fail;}
    return s;
fail:
    freeobs(&dummy);qr_error(error,error_size,why?why:"open failed");qr_close(s);return NULL;
}
EXPORT int qr_restart(void *session,char *error,int error_size)
{
    qr_session *s=(qr_session*)session;
    if(!s) return qr_error(error,error_size,"session is closed");
    s->cursor=0;s->previous.time=0;s->previous.sec=0;s->failed=0;
    if(!qr_new_filter(s)) return qr_error(error,error_size,"cannot reset filter");
    return 0;
}
EXPORT int qr_step(void *session,qr_callback callback,void *user,char *error,int error_size)
{
    qr_session *s=(qr_session*)session;
    qr_rtk_context *c;
    qr_event event;
    obsd_t obs[2*MAXOBS];
    int lo=0,hi,mid,k,nr,nb,r0,b0,i,valid_pair,ret=0;
    double age,meta[4];
    gtime_t t;
    if(!s || s->failed) return qr_error(error,error_size,"session failed or closed; restart before reuse");
    if(s->cursor>=s->nr) return 0;
    if(s->mode && !s->loaded && !callback) return qr_error(error,error_size,"Q/R mode requires a model or training callback");
    r0=s->ri[s->cursor];nr=s->ri[s->cursor+1]-r0;t=s->rover.data[r0].time;
    hi=s->nb;
    while(lo<hi) {
        mid=(lo+hi)/2;
        if(timediff(s->base.data[s->bi[mid]].time,t)<0) lo=mid+1; else hi=mid;
    }
    k=lo<s->nb?lo:s->nb-1;
    if(k>0 && fabs(timediff(s->base.data[s->bi[k-1]].time,t))<
              fabs(timediff(s->base.data[s->bi[k]].time,t))) k--;
    b0=s->bi[k];nb=s->bi[k+1]-b0;
    age=timediff(t,s->base.data[b0].time);valid_pair=fabs(age)<=s->pair_age+1e-9;
    if(nr>MAXOBS || nb>MAXOBS) {s->failed=1;return qr_error(error,error_size,"epoch exceeds compiled MAXOBS");}
    if(valid_pair && s->previous.time && timediff(t,s->previous)>s->max_gap) {
        if(!qr_new_filter(s)) {s->failed=1;return qr_error(error,error_size,"gap reset allocation failed");}
    }
    c=(qr_rtk_context*)s->rtk.learned_qr;c->callback=callback;c->user=user;
    memset(&event,0,sizeof(event));event.kind=QR_BEGIN;event.nx=s->rtk.nx;
    event.time=qr_seconds(t);event.n=s->first;event.m=NFREQ;
    event.x=s->rtk.x;event.P=s->rtk.P;qr_notify(c,&event);s->first=0;
    if(!c->failed && valid_pair) {
        for(i=0;i<nr;i++) obs[i]=s->rover.data[r0+i];
        for(i=0;i<nb;i++) obs[nr+i]=s->base.data[b0+i];
        ret=rtkpos(&s->rtk,obs,nr+nb,&s->nav);s->previous=t;
    }
    event.kind=QR_END;event.n=valid_pair?s->rtk.sol.stat:SOLQ_NONE;
    event.m=valid_pair?s->rtk.sol.ns:0;event.a=s->rtk.sol.rr;
    meta[0]=s->rtk.sol.ratio;meta[1]=age;meta[2]=ret;meta[3]=s->cursor;
    event.b=meta;qr_notify(c,&event);s->cursor++;
    if(c->failed) {s->failed=1;return qr_error(error,error_size,"training callback failed; session aborted");}
    return 1;
}
