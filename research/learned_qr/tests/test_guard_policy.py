import ctypes as C
import inspect
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
import qrlearn.model as M


def hist(cn0=30.,code=True,base=45.):
    h=torch.zeros(10,13,dtype=torch.float64);h[:,-1]=1
    h[:,0]=(cn0-40)/10;h[:,1]=(base-40)/10;h[:,3]=int(code)
    return h


def model(cap=1.):
    assert hasattr(M,'GuardedCodeNet'), 'guard policy missing'
    return M.QRModel(r_policy='code_guard',phase_guard_cap=cap).double()


def test_identity_learns_but_no_shrinking_or_clean_reweighting():
    m=model();h=hist();assert m.r(h).item()==1
    m.r(h).backward();assert m.r.fc2.bias.grad.abs().sum()>0
    with torch.no_grad():m.r.fc2.bias.fill_(.4)
    for h in (hist(code=False),hist(45),hist(0,base=0)):
        assert m.r(h).item()==1
    assert m.r(hist()).item()>1
    with torch.no_grad():m.r.fc2.bias.fill_(-.4)
    assert m.r(hist()).item()==1


def test_phase_guard_and_checkpoint():
    m=model(4.)
    assert m.r(hist(30,False)).item()==4
    assert m.r(hist(35,False)).item()==2.5
    assert m.r(hist(45,False)).item()==1
    m.r(hist(30,False)).backward();assert m.r.fc2.bias.grad.abs().sum()==0
    with pytest.raises(ValueError):model(0)


def test_position_model_numerical_gradient():
    m=model()
    with torch.no_grad():m.r.fc2.bias.fill_(.2)
    m.r(hist()).backward();grad=m.r.fc2.bias.grad.item();eps=1e-6
    with torch.no_grad():
        m.r.fc2.bias.add_(eps);a=m.r(hist()).item()
        m.r.fc2.bias.sub_(2*eps);b=m.r(hist()).item()
    assert grad==pytest.approx((a-b)/(2*eps),rel=1e-6)


@pytest.mark.parametrize('cap',[1.,4.])
def test_export_c_pytorch_parity_and_header_validation(tmp_path,cap):
    m=model(cap);torch.manual_seed(19)
    with torch.no_grad():
        m.r.fc2.weight.normal_(0,.15);m.r.fc2.bias.fill_(.4)
        m.q.fc2.weight.normal_(0,.05)
    p=tmp_path/'m.qr';m.export(p,provenance='synthetic')
    c=tmp_path/'wrap.c';c.write_text('''#include "learned_qr.h"
int evaluate(const char *p,int q,const double *x,double *y){qr_model m;
if(!qr_load(&m,p,1)) return 0;return qr_model_forward(&m,q,x,y);}''')
    include=Path(__file__).resolve().parents[1]/'src'
    subprocess.run(['cc','-std=c99','-shared','-fPIC','-O2','-Wall','-Wextra',
        '-I'+str(include),str(c),'-lm','-o',str(tmp_path/'w.so')],check=True)
    lib=C.CDLL(str(tmp_path/'w.so'));D=C.POINTER(C.c_double)
    lib.evaluate.argtypes=[C.c_char_p,C.c_int,D,D]
    for code in (False,True):
        for cn0 in (0.,25.,35.,40.,50.):
            h=hist(cn0,code);x=h.numpy();out=np.zeros(2)
            assert lib.evaluate(str(p).encode(),0,x.ctypes.data_as(D),out.ctypes.data_as(D))==1
            assert out[0]==pytest.approx(m.r(h).item(),rel=3e-12,abs=3e-12)
    q=np.random.default_rng(10).normal(size=(10,11));out=np.zeros(2)
    assert lib.evaluate(str(p).encode(),1,q.ctypes.data_as(D),out.ctypes.data_as(D))==1
    np.testing.assert_allclose(out,m.q(torch.tensor(q)).detach().numpy(),rtol=3e-12,atol=3e-12)
    good=p.read_text();p.write_text(good.replace(f'1 40 10 {cap:g}\n',f'1 nan 10 {cap:g}\n'))
    assert lib.evaluate(str(p).encode(),0,x.ctypes.data_as(D),out.ctypes.data_as(D))==0


def test_checkpoint_roundtrip(tmp_path):
    from qrlearn.training import checkpoint_save,checkpoint_load
    m=model(4);p=tmp_path/'m.pt';checkpoint_save(p,m,max_gap=30,epoch=1,mode='qr')
    other,_=checkpoint_load(p)
    assert other.r_policy=='code_guard' and other.r.phase_cap==4
    torch.testing.assert_close(other.r(hist()),m.r(hist()))
