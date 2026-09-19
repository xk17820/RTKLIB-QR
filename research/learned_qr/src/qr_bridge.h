/* Local raw-RINEX bridge. No reference coordinates are accepted by this API. */
#ifndef RTKLIB_QR_BRIDGE_H
#define RTKLIB_QR_BRIDGE_H
#include "rtklib.h"
#include "qr_trace.h"
#ifdef __cplusplus
extern "C" {
#endif
EXPORT int qr_bridge_abi(void);
EXPORT int qr_set_stable(void *session,int enabled);
EXPORT void *qr_open(const char *config,const char *rover,const char *base,
                     const char **navfiles,int nnav,const char *model,int mode,
                     int training,int allow_test,double max_gap,double pair_age,
                     char *error,int error_size);
EXPORT int qr_step(void *session,qr_callback callback,void *user,
                   char *error,int error_size);
EXPORT int qr_restart(void *session,char *error,int error_size);
EXPORT void qr_close(void *session);
#ifdef __cplusplus
}
#endif
#endif
