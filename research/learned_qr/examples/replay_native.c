/* C-only inference/replay using the SAME session/gap policy as the trainer.
 * No Python, PyTorch, reference positions or gradient runtime is required.
 * Linux build command is in docs/TRAINING.md. Rebuild against this branch's ABI.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "qr_bridge.h"

static int output_epoch(void *user,const qr_event *event)
{
    FILE *out=(FILE*)user;
    if(event->kind==QR_END) {
        const double *p=event->a;
        double x=NAN,y=NAN,z=NAN;
        if(event->n>0) {x=p[0];y=p[1];z=p[2];}
        if(fprintf(out,"%.9f,%.17g,%.17g,%.17g,%d,%d,%.9g,%.9g\n",
                   event->time,x,y,z,event->n,event->m,event->b[0],event->b[1])<0) return 1;
    }
    return 0;
}
int main(int argc,char **argv)
{
    char error[512],*end;double gap;void *session;FILE *out;int status,mode;
    if(argc<8) {
        fprintf(stderr,"Usage: %s config rover.obs base.obs model.qr|- output.csv max_gap nav1 [nav2 ...]\n",argv[0]);
        return 2;
    }
    gap=strtod(argv[6],&end);
    if(*end || !isfinite(gap) || gap<=0) {fprintf(stderr,"Invalid max_gap\n");return 2;}
    mode=strcmp(argv[4],"-")?3:0;
    session=qr_open(argv[1],argv[2],argv[3],(const char**)&argv[7],argc-7,
                    mode?argv[4]:NULL,mode,0,0,gap,.05,error,sizeof(error));
    if(!session) {fprintf(stderr,"%s\n",error);return 2;}
    /* 'x' means exclusive creation on supported C runtimes: never overwrite. */
    out=fopen(argv[5],"wx");
    if(!out) {perror(argv[5]);qr_close(session);return 2;}
    fprintf(out,"t_gpst_s,x_m,y_m,z_m,status,ns,ratio,age_s\n");
    while((status=qr_step(session,output_epoch,out,error,sizeof(error)))>0) {}
    if(status<0) fprintf(stderr,"%s\n",error);
    if(fclose(out)) status=-1;
    qr_close(session);
    return status<0?2:0;
}
