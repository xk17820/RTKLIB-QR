"""A code history containing dual-frequency MP must reset on either input."""
import ctypes as C
import subprocess
from pathlib import Path
import pytest
from test_adapter import STUB


@pytest.fixture(scope='module')
def evidence_lib(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    temp = tmp_path_factory.mktemp('scl-history')
    code = STUB + r'''
    int mp_history_count(int policy,int frq,int event,double *anomaly) {
      rtk_t *r=(rtk_t*)calloc(1,sizeof(*r));
      r->opt.mode=2;r->opt.nf=2;r->tt=1;r->nx=9;
      qr_rtk_context *ctx=(qr_rtk_context*)calloc(1,sizeof(*ctx));
      r->learned_qr=ctx;ctx->model.max_gap=30;ctx->model.rpolicy=policy;
      ctx->r=(qr_history*)calloc(MAXSAT*2*NFREQ,sizeof(qr_history));
      obsd_t o={0},b={0};int other=1-frq,t,result;
      o.code[0]=b.code[0]=1;o.code[1]=b.code[1]=2;
      o.P[0]=o.P[1]=21000100;b.P[0]=b.P[1]=21000000;
      o.L[0]=110000100;b.L[0]=110000000;
      o.L[1]=90000100;b.L[1]=90000000;
      for(t=1;t<=2;t++) {
        o.time.time=t;
        qr_apply_r(r,1,SYS_GPS,.7,45,45,1000,0,2+frq,&r->opt,&o,&b,4.);
      }
      o.time.time=3;o.L[other]+=8;
      if(event==1)o.LLI[other]=LLI_SLIP;
      if(event==2)b.LLI[other]=LLI_SLIP;
      if(event==3)r->ssat[0].slip[other]=LLI_SLIP;
      if(event==4)o.code[other]+=1;
      if(event==5)b.code[other]+=1;
      if(event==6)o.time.time=40;
      qr_apply_r(r,1,SYS_GPS,.7,45,45,1000,0,2+frq,&r->opt,&o,&b,4.);
      /* A repeated native residual evaluation must not advance history. */
      qr_apply_r(r,1,SYS_GPS,.7,45,45,1000,0,2+frq,&r->opt,&o,&b,4.);
      qr_history *h=&ctx->r[NFREQ+frq];
      result=h->count;*anomaly=h->values[(h->next+QR_WINDOW-1)%QR_WINDOW][9];
      qr_rtk_destroy(ctx);free(r);return result;
    }
    '''
    src = temp/'check.c'; src.write_text(code)
    so = temp/'check.so'
    subprocess.run(['cc','-std=c99','-shared','-fPIC','-Wall','-Wextra','-Werror',
                    '-I'+str(root/'src'),str(src),'-lm','-o',str(so)],check=True)
    lib = C.CDLL(str(so))
    lib.mp_history_count.argtypes=[C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double)]
    lib.mp_history_count.restype=C.c_int
    return lib


@pytest.mark.parametrize('policy',[2,3])
@pytest.mark.parametrize('frq',[0,1])
@pytest.mark.parametrize('event',[1,2,3,4,5,6])
def test_all_mp_dependencies_reset_history_not_only_current_anomaly(evidence_lib,policy,frq,event):
    anomaly=C.c_double()
    count=evidence_lib.mp_history_count(policy,frq,event,C.byref(anomaly))
    assert abs(anomaly.value)<1e-12
    assert count==1, 'cross-frequency discontinuity must clear the temporal MP input'


@pytest.mark.parametrize('policy',[2,3])
@pytest.mark.parametrize('frq',[0,1])
def test_unmarked_phase_change_stays_as_evidence_not_an_automatic_reset(evidence_lib,policy,frq):
    anomaly=C.c_double()
    assert evidence_lib.mp_history_count(policy,frq,0,C.byref(anomaly))==3
    assert abs(anomaly.value)>.1


@pytest.mark.parametrize('policy',[0,1])
def test_legacy_features_keep_single_frequency_history_contract(evidence_lib,policy):
    anomaly=C.c_double()
    assert evidence_lib.mp_history_count(policy,0,1,C.byref(anomaly))==3
