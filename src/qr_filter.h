/* Numerically stable Kalman update for learned runs. The legacy off path still
 * calls upstream filter(). Compare learned Q/R to RTKLIB_QR_STABLE=1 baseline.
 * Cholesky solves avoid an explicit inverse; Joseph form preserves covariance.
 * No jitter, eigenvalue clipping, ambiguity rounding or acceptance changes.
 */
#ifndef RTKLIB_QR_FILTER_H
#define RTKLIB_QR_FILTER_H
static int qr_stable_filter(double *x,double *P,const double *H,const double *v,
                            const double *R,int n,int m)
{
    int i,j,k,a,b,na=0,*ix=NULL,info=-1;
    double *Pa=NULL,*Ha=NULL,*PH=NULL,*S=NULL,*K=NULL,*A=NULL,*AP=NULL,*C=NULL,*KR=NULL;
    double t;
    ix=imat(n,1);
    for(i=0;i<n;i++) if(x[i]!=0.0 && P[i+i*n]>0.0) ix[na++]=i;
    if(!na || !m) goto end;
    Pa=mat(na,na);Ha=mat(na,m);PH=mat(na,m);S=mat(m,m);K=mat(na,m);
    A=eye(na);AP=mat(na,na);C=mat(na,na);KR=mat(na,m);
    for(i=0;i<na;i++) {
        for(j=0;j<na;j++) Pa[i+j*na]=0.5*(P[ix[i]+ix[j]*n]+P[ix[j]+ix[i]*n]);
        for(j=0;j<m;j++) Ha[i+j*na]=H[ix[i]+j*n];
    }
    matmul("NN",na,m,na,Pa,Ha,PH);
    matcpy(S,R,m,m);matmulp("TN",m,m,na,Ha,PH,S);
    /* Lower triangular Cholesky. Upper entries are no longer needed. */
    for(i=0;i<m;i++) for(j=0;j<=i;j++) {
        t=0.5*(S[i+j*m]+S[j+i*m]);
        for(k=0;k<j;k++) t-=S[i+k*m]*S[j+k*m];
        if(i==j) {
            if(!isfinite(t) || t<=0) {info=-2;goto end;}
            S[i+j*m]=sqrt(t);
        } else S[i+j*m]=t/S[j+j*m];
    }
    /* K row by row: L y = PH', L' k = y. */
    for(a=0;a<na;a++) {
        for(i=0;i<m;i++) {
            t=PH[a+i*na];for(j=0;j<i;j++) t-=S[i+j*m]*K[a+j*na];
            K[a+i*na]=t/S[i+i*m];
        }
        for(i=m-1;i>=0;i--) {
            t=K[a+i*na];for(j=i+1;j<m;j++) t-=S[j+i*m]*K[a+j*na];
            K[a+i*na]=t/S[i+i*m];
        }
    }
    matmulm("NT",na,na,m,K,Ha,A);
    matmul("NN",na,na,na,A,Pa,AP);matmul("NT",na,na,na,AP,A,C);
    matmul("NN",na,m,m,K,R,KR);matmulp("NT",na,na,m,KR,K,C);
    for(a=0;a<na;a++) {
        t=0;for(b=0;b<m;b++) t+=K[a+b*na]*v[b];
        if(!isfinite(t)) {info=-3;goto end;}
        AP[a]=t;
        for(b=0;b<na;b++) if(!isfinite(C[a+b*na])) {info=-3;goto end;}
    }
    for(a=0;a<na;a++) {
        x[ix[a]]+=AP[a];
        for(b=0;b<na;b++) P[ix[a]+ix[b]*n]=0.5*(C[a+b*na]+C[b+a*na]);
    }
    info=0;
end:
    free(ix);free(Pa);free(Ha);free(PH);free(S);free(K);free(A);free(AP);free(C);free(KR);
    return info;
}
#endif
