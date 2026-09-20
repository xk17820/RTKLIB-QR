/* Stable callback ABI for conditional first-order FLOAT differentiation.
 * Arrays are borrowed for this call ONLY. Matrices use RTKLIB column-major order.
 * Only INFER_Q/INFER_R.b is writable (positive variance multipliers).
 * Returning nonzero aborts the session at the epoch boundary.
 */
#ifndef RTKLIB_QR_TRACE_H
#define RTKLIB_QR_TRACE_H

enum {
    QR_BEGIN=1, QR_INIT=2, QR_TRANSITION=3, QR_QADD=4,
    QR_INFER_Q=5, QR_INFER_R=6, QR_SD_VARIANCE=7, QR_DIAG=8,
    QR_ZERO=9, QR_OFFSET=10, QR_PRIOR=11, QR_FILTER=12,
    QR_ACCEPT=13, QR_END=14, QR_CANDIDATE=15, QR_FIXED_RESULT=16
};
typedef struct {
    int kind, nx, n, m;
    double time;
    const int *ix;
    const double *x, *P, *a;
    double *b;
    const double *c;
} qr_event;
typedef int (*qr_callback)(void *user, const qr_event *event);
#endif
