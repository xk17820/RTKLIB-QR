from pathlib import Path
import ctypes as c
import subprocess
import math
import os
import pytest
import torch
from qrlearn.model import QRModel

ROOT=Path(__file__).resolve().parents[1]
STUB=r'''
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define NFREQ 3
#define FREQL1 1575.42e6
#define FREQL2 1227.60e6
#define CLIGHT 299792458.0
#define MAXSAT 8
#define PMODE_DGPS 1
#define PMODE_KINEMA 2
#define PMODE_FIXED 5
#define IONOOPT_IFLC 3
#define LLI_SLIP 1
#define LLI_HALFC 2
#define SYS_GPS 1
#define SYS_GLO 4
#define SYS_GAL 8
#define SYS_SBS 2
#define SYS_QZS 16
#define SYS_CMP 32
#define SYS_IRN 64
typedef struct {long time;double sec;} gtime_t;
typedef struct {gtime_t time;unsigned char code[3],LLI[3];float Pstd[3],Lstd[3]; double P[3],L[3],D[3];} obsd_t;
typedef struct {int mode,dynamics,ionoopt,nf; double eratio[3],err[8];} prcopt_t;
typedef struct {int sys; unsigned char slip[3];} ssat_t;
typedef struct {gtime_t time;int ns,stat;} sol_t;
typedef struct {void *learned_qr;prcopt_t opt;sol_t sol;double *x,*P,tt;int nx;ssat_t ssat[MAXSAT];} rtk_t;
static double time2gpst(gtime_t t,int*w) {*w=0;return t.time+t.sec;}
static void ecef2pos(const double*x,double*p) {(void)x; p[0]=p[1]=p[2]=0;}
static void ecef2enu(const double*p,const double*x,double*v) {(void)p;memcpy(v,x,3*sizeof(double));}
#include "learned_qr_adapter.h"
void *new_rtk(void) {
 rtk_t *r=(rtk_t*)calloc(1,sizeof(*r));r->opt.mode=2;r->opt.dynamics=1;r->opt.nf=3;
 r->tt=1;r->nx=9;r->x=(double*)calloc(9,sizeof(double));r->P=(double*)calloc(81,sizeof(double));
 r->sol.time.time=10;r->sol.ns=20;r->sol.stat=2;r->learned_qr=qr_rtk_create();return r;
}
void del_rtk(rtk_t*r){qr_rtk_destroy(r->learned_qr);free(r->x);free(r->P);free(r);}
int ctx_exists(rtk_t*r){return r->learned_qr!=NULL;}
double rvar(rtk_t*r,double t,int slip,int basecode,double variance) {
 obsd_t o={0},b={0};o.time.time=(long)t;o.time.sec=t-(long)t;o.code[0]=1;
 b=o;b.code[0]=(unsigned char)basecode;b.LLI[0]=(unsigned char)slip;
 return qr_apply_r(r,1,SYS_GPS,.7,42,43,1000,0,0,&r->opt,&o,&b,variance);
}
int seen(rtk_t*r){qr_rtk_context*p=(qr_rtk_context*)r->learned_qr;return p?p->r[0].seen:0;}
void qvar(rtk_t*r,double*out){double Q[9]={1,0,0,0,1,0,0,0,2};qr_apply_q(r,1,Q);out[0]=Q[0];out[1]=Q[4];out[2]=Q[8];}
void backward(rtk_t*r){r->tt=-1;}
'''
@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    assert (ROOT/'src/learned_qr_adapter.h').exists(),'RTK adapter must be implemented'
    p=tmp_path_factory.mktemp('adapter');(p/'stub.c').write_text(STUB)
    subprocess.run(['cc','-std=c99','-shared','-fPIC','-Wall','-Wextra','-Werror','-I'+str(ROOT/'src'),str(p/'stub.c'),'-lm','-o',str(p/'a.so')],check=True)
    lib=c.CDLL(str(p/'a.so'));v=c.c_void_p
    lib.new_rtk.restype=v;lib.del_rtk.argtypes=[v];lib.ctx_exists.argtypes=[v]
    lib.rvar.argtypes=[v,c.c_double,c.c_int,c.c_int,c.c_double];lib.rvar.restype=c.c_double
    lib.seen.argtypes=[v];lib.qvar.argtypes=[v,c.POINTER(c.c_double)];lib.backward.argtypes=[v]
    return lib
@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ('RTKLIB_QR_MODE','RTKLIB_QR_MODEL','RTKLIB_QR_LOG','RTKLIB_QR_ALLOW_TEST_MODEL'):monkeypatch.delenv(k,raising=False)

def configure(tmp_path,monkeypatch,mode='qr'):
    net=QRModel().double()
    with torch.no_grad():net.q.fc2.bias.fill_(.2);net.r.fc2.bias.fill_(.3)
    p=tmp_path/'m.qr';net.export(p,provenance='synthetic')
    monkeypatch.setenv('RTKLIB_QR_MODE',mode);monkeypatch.setenv('RTKLIB_QR_MODEL',str(p))
    monkeypatch.setenv('RTKLIB_QR_ALLOW_TEST_MODEL','1')

def test_default_disabled_is_exact_identity(lib):
    r=lib.new_rtk()
    try: assert not lib.ctx_exists(r); assert lib.rvar(r,1,0,1,.123456789)==.123456789
    finally:lib.del_rtk(r)

def test_r_scaling_caching_slip_base_signal_and_context_isolation(lib,tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch);a,b=lib.new_rtk(),lib.new_rtk()
    try:
        assert lib.rvar(a,1,0,1,2.)==pytest.approx(2*math.exp(math.log(1000)*math.tanh(.3)))
        lib.rvar(a,1,1,1,2);assert lib.seen(a)==1 and lib.seen(b)==0
        lib.rvar(a,2,0,1,2);assert lib.seen(a)==2
        lib.rvar(a,3,1,1,2);assert lib.seen(a)==1
        lib.rvar(a,4,0,2,2);assert lib.seen(a)==1
        lib.rvar(a,40,0,2,2);assert lib.seen(a)==1
        lib.backward(a);assert lib.rvar(a,41,0,2,2)==2
    finally:lib.del_rtk(a);lib.del_rtk(b)

def test_q_scales_only_existing_horizontal_vertical_terms(lib,tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch);r=lib.new_rtk();out=(c.c_double*3)()
    try:
        lib.qvar(r,out);factor=math.exp(math.log(10)*math.tanh(.2))
        assert list(out)==pytest.approx([factor,factor,2*factor])
    finally:lib.del_rtk(r)

def test_ablation_q_only_and_r_only(lib,tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch,'q');r=lib.new_rtk()
    try:assert lib.rvar(r,1,0,1,3)==3
    finally:lib.del_rtk(r)
    monkeypatch.setenv('RTKLIB_QR_MODE','r');r=lib.new_rtk();out=(c.c_double*3)()
    try:lib.qvar(r,out);assert list(out)==[1,1,2]
    finally:lib.del_rtk(r)

def test_invalid_model_falls_back_and_optional_log_records_features(lib,tmp_path,monkeypatch):
    monkeypatch.setenv('RTKLIB_QR_MODE','qr');monkeypatch.setenv('RTKLIB_QR_MODEL','/missing.qr')
    log=tmp_path/'log.csv';monkeypatch.setenv('RTKLIB_QR_LOG',str(log))
    r=lib.new_rtk()
    try:assert lib.rvar(r,1,0,1,2)==2
    finally:lib.del_rtk(r)
    text=log.read_text();assert 't_gpst_s' in text and '\nR,' in text
