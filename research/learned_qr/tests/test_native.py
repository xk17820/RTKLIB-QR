import ctypes as c
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from qrlearn.model import QRModel

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=r'''
#include "learned_qr.h"
qr_model *test_load(const char *p,int allow) {
    qr_model *m=(qr_model*)calloc(1,sizeof(*m));
    if (!m || !qr_load(m,p,allow)) { free(m); return NULL; } return m;
}
void test_free(void *m) {free(m);}
void test_eval(qr_model *m,int isq,const double *x,double *out) {
    qr_forward(isq?&m->q:&m->r,x,isq?m->qlimit:m->rlimit,out);
}
qr_history *test_hist(void) { return (qr_history*)calloc(1,sizeof(qr_history)); }
void test_step(qr_history *h,double t,int sig,int reset,const double *x,int d,double gap) {
    qr_history_push(h,t,(unsigned)sig,reset,x,d,gap);
}
void test_flat(qr_history *h,int d,double *out) {qr_history_flat(h,d,out);}
int test_count(qr_history *h) { return h->count; }
'''

@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    assert (ROOT/'src/learned_qr.h').exists(), 'native runtime must be implemented'
    d=tmp_path_factory.mktemp('native'); (d/'wrap.c').write_text(WRAPPER)
    subprocess.run(['cc','-std=c99','-shared','-fPIC','-O2','-Wall','-Wextra','-Werror',
                    '-I'+str(ROOT/'src'),str(d/'wrap.c'),'-lm','-o',str(d/'libqr.so')],check=True)
    lib=c.CDLL(str(d/'libqr.so')); ptr=c.c_void_p; arr=c.POINTER(c.c_double)
    lib.test_load.argtypes=[c.c_char_p,c.c_int];lib.test_load.restype=ptr
    lib.test_free.argtypes=[ptr]
    lib.test_eval.argtypes=[ptr,c.c_int,arr,arr]
    lib.test_hist.restype=ptr
    lib.test_step.argtypes=[ptr,c.c_double,c.c_int,c.c_int,arr,c.c_int,c.c_double]
    lib.test_flat.argtypes=[ptr,c.c_int,arr];lib.test_count.argtypes=[ptr]
    return lib

def ptr(a): return np.ascontiguousarray(a,dtype=np.float64).ctypes.data_as(c.POINTER(c.c_double))

def test_native_matches_nontrivial_torch_network(lib,tmp_path):
    torch.manual_seed(7); net=QRModel().double()
    with torch.no_grad():
        for branch in (net.q,net.r):
            branch.fc2.weight.normal_(0,.2);branch.fc2.bias.normal_(0,.1)
    p=tmp_path/'network.qr';net.export(p,provenance='synthetic')
    m=lib.test_load(str(p).encode(),1); assert m
    try:
        rng=np.random.default_rng(8)
        for qflag,d,branch in ((1,11,net.q),(0,13,net.r)):
            for _ in range(15):
                x=rng.normal(size=(10,d)); out=np.zeros(branch.outputs)
                lib.test_eval(m,qflag,ptr(x),ptr(out))
                expected=branch(torch.from_numpy(x)).detach().numpy()
                np.testing.assert_allclose(out,expected,rtol=3e-12,atol=3e-12)
    finally:lib.test_free(m)

def test_native_refuses_untrained_and_synthetic_by_default(lib,tmp_path):
    p=tmp_path/'test.qr'
    for provenance in ('untrained','synthetic'):
        QRModel().double().export(p,provenance=provenance)
        assert not lib.test_load(str(p).encode(),0)
    assert not lib.test_load(b'/nonexistent/model.qr',0)

def test_native_rejects_truncated_nan_extra_and_wrong_dimensions(lib,tmp_path):
    p=tmp_path/'bad.qr'; QRModel().double().export(p,provenance='synthetic')
    text=p.read_text()
    for bad in (text[:100],text.replace('110 2','111 2'),text+'EXTRA',text.replace(' 10 ',' nan ',1)):
        p.write_text(bad); assert not lib.test_load(str(p).encode(),1)

def test_history_deduplicates_resets_and_zero_pads(lib):
    h=lib.test_hist(); x=np.arange(10,dtype=float); out=np.zeros((10,11))
    try:
        lib.test_step(h,1,5,0,ptr(x),10,30)
        lib.test_step(h,1,5,1,ptr(x+1),10,30)
        assert lib.test_count(h)==1
        lib.test_flat(h,10,ptr(out)); np.testing.assert_array_equal(out[-1,:10],x)
        assert out[:-1].sum()==0 and out[-1,-1]==1
        lib.test_step(h,2,5,0,ptr(x+1),10,30); assert lib.test_count(h)==2
        lib.test_step(h,3,6,0,ptr(x),10,30); assert lib.test_count(h)==1
        lib.test_step(h,4,6,1,ptr(x),10,30); assert lib.test_count(h)==1
        lib.test_step(h,99,6,0,ptr(x),10,30); assert lib.test_count(h)==1
        lib.test_step(h,98,6,0,ptr(x),10,30); assert lib.test_count(h)==1
    finally:lib.test_free(h)

def test_histories_are_isolated_and_roll_oldest_first(lib):
    a,b=lib.test_hist(),lib.test_hist(); out=np.zeros((10,11))
    try:
        for i in range(15):lib.test_step(a,float(i+1),1,0,ptr(np.full(10,i,dtype=float)),10,30)
        assert lib.test_count(a)==10 and lib.test_count(b)==0
        lib.test_flat(a,10,ptr(out)); np.testing.assert_array_equal(out[:,0],np.arange(5,15))
    finally:lib.test_free(a);lib.test_free(b)
