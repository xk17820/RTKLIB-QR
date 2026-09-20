from pathlib import Path
import ctypes as ct
import numpy as np
import pytest
import torch
from qrlearn.model import QRModel
from test_live import session_kwargs,ROOT,KIT


def test_conditional_objective_has_correct_fixed_candidate_gradient():
    import importlib.util
    assert importlib.util.find_spec('qrlearn.scl') is not None,'SCL conditional objective not implemented'
    from qrlearn.scl import condition_position
    torch.manual_seed(11)
    x=torch.tensor([1.,2.,3.,10.2,15.8,20.4],dtype=torch.float64,requires_grad=True)
    L=torch.randn(6,6,dtype=torch.float64);P=(L@L.T+torch.eye(6)).requires_grad_()
    pairs=torch.tensor([[3,4],[3,5]]);a=torch.tensor([-6.,-10.],dtype=torch.float64,requires_grad=True)
    def f(x,P):return condition_position(x,P,pairs,a)
    assert torch.autograd.gradcheck(f,(x,P),eps=1e-6,atol=1e-5,rtol=1e-4)
    f(x,P).square().sum().backward()
    assert a.grad is None, 'integer search output must be detached'
    assert x.grad.abs().sum()>0 and P.grad.abs().sum()>0


def test_scl_evidence_not_cn0_controls_inflation():
    from qrlearn.model import QRModel
    m=QRModel(r_policy='scl').double()
    with torch.no_grad():m.r.fc2.bias.fill_(.15)
    h=torch.zeros(10,13,dtype=torch.float64);h[-1,-1]=1;h[-1,3]=1
    # V3 f8/f9 are consistency evidence, never a reference-derived label.
    h[-1,8]=np.log1p(12.)
    hi=h.clone();hi[-1,:2]=1.2
    lo=h.clone();lo[-1,:2]=-1.2
    assert m.r(hi).item()>1 and m.r(lo).item()>1
    clean=lo.clone();clean[-1,8:10]=0
    assert m.r(clean).item()==1.
    phase=hi.clone();phase[-1,3]=0
    assert m.r(phase).item()==1.


def test_scl_native_export_agrees_with_torch(tmp_path):
    from qrlearn.scl import CompactTrace
    from qrlearn.native import Session
    m=QRModel(r_policy='scl').double()
    with torch.no_grad():m.q.fc2.bias.fill_(.01);m.r.fc2.bias.fill_(.2)
    p=tmp_path/'m.qr';m.export(p,provenance='synthetic',max_gap=31)
    eng=CompactTrace(m);a=[];b=[]
    with Session(**session_kwargs(),mode='qr',training=True,scl=True,candidates=True) as s:
        for _ in range(12):a.append(s.step(eng.handle)['position'])
    with Session(**session_kwargs(),mode='qr',training=True,model=p,allow_test=True) as s:
        for _ in range(12):b.append(s.step()['position'])
    np.testing.assert_allclose(a,b,atol=2e-5,rtol=0)
    assert eng.max_state_discrepancy<.001


def test_candidate_observer_does_not_modify_native_solution():
    from qrlearn.native import Session,array
    got=[];a=[];b=[]
    def callback(e):
        if e.kind==15:
            pairs=array(e.ix,2*e.n).reshape(e.n,2)
            x=array(e.x,e.nx);P=array(e.P,e.nx*e.nx).reshape(e.nx,e.nx,order='F')
            D=np.zeros((e.n,e.nx));D[np.arange(e.n),pairs[:,0]]=1;D[np.arange(e.n),pairs[:,1]]=-1
            ints=array(e.a,e.n);pos=x[:3]-P[:3]@D.T@np.linalg.solve(D@P@D.T,D@x-ints)
            np.testing.assert_allclose(pos,array(e.b,3),atol=1e-5,rtol=0);got.append(ints)
    with Session(**session_kwargs(),training=True,mode='off',candidates=True) as s:
        for _ in range(15):a.append(s.step(callback)['position'])
    with Session(**session_kwargs(),training=True,mode='off') as s:
        for _ in range(15):b.append(s.step()['position'])
    np.testing.assert_array_equal(a,b)
    assert len(got)>3


def test_compact_tape_matches_dense_and_aux_loss_backprop():
    from qrlearn.scl import CompactTrace
    from qrlearn.native import Session
    from qrlearn.live import TraceEngine
    torch.manual_seed(8);m=QRModel().double()
    with torch.no_grad():m.r.fc2.bias.fill_(.025);m.q.fc2.bias.fill_(.015)
    dense=TraceEngine(m);compact=CompactTrace(m)
    for engine in (dense,compact):
        with Session(**session_kwargs(),training=True,mode='qr',candidates=isinstance(engine,CompactTrace)) as s:
            for _ in range(12):s.step(engine.handle)
    ids=compact.global_ids
    torch.testing.assert_close(compact.x,dense.x[ids],atol=1e-5,rtol=0)
    torch.testing.assert_close(compact.P,dense.P[ids[:,None],ids[None,:]],atol=1e-4,rtol=0)
    assert len(ids)<100 and compact.fixed_position is not None
    loss=compact.position.square().sum()*1e-7+compact.fixed_position.square().sum()*1e-7
    loss.backward()
    for net in (m.q,m.r):assert sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)>0


def test_mid_session_model_reload_keeps_state(tmp_path):
    from qrlearn.native import Session
    model=QRModel(r_policy='scl').double();p=tmp_path/'identity.qr';model.export(p,provenance='synthetic',max_gap=31)
    a=[];b=[]
    with Session(**session_kwargs(),training=True,mode='qr',model=p,allow_test=True) as s:
        assert hasattr(s,'reload_model'),'state-preserving parameter reload not implemented'
        for k in range(16):
            if k==7:s.reload_model(p,allow_test=True)
            a.append(s.step()['position'])
    with Session(**session_kwargs(),training=True,mode='qr',model=p,allow_test=True) as s:
        for k in range(16):b.append(s.step()['position'])
    np.testing.assert_array_equal(a,b)


def test_mp_anchor_resets_when_other_frequency_slips(tmp_path):
    """L1 multipath combination contains L2; L2-only discontinuities must reset it."""
    import subprocess
    from test_adapter import STUB
    stub=STUB+r'''
    double evidence_check(int other_slip) {
      rtk_t *r=(rtk_t*)calloc(1,sizeof(*r));
      r->opt.mode=2;r->opt.nf=2;r->tt=1;r->nx=9;
      r->learned_qr=calloc(1,sizeof(qr_rtk_context));
      qr_rtk_context *ctx=(qr_rtk_context*)r->learned_qr;ctx->model.max_gap=30;ctx->model.rpolicy=2;ctx->r=calloc(MAXSAT*2*NFREQ,sizeof(qr_history));
      obsd_t o={0},b={0};o.code[0]=b.code[0]=1;o.code[1]=b.code[1]=2;
      o.P[0]=21000100;b.P[0]=21000000;o.L[0]=110000100;b.L[0]=110000000;o.L[1]=90000100;b.L[1]=90000000;
      o.time.time=1;qr_apply_r(r,1,SYS_GPS,.7,45,45,1000,0,2,&r->opt,&o,&b,4.);
      o.time.time=2;o.L[1]+=8;o.LLI[1]=other_slip;
      qr_apply_r(r,1,SYS_GPS,.7,45,45,1000,0,2,&r->opt,&o,&b,4.);
      qr_history *h=&ctx->r[NFREQ];double value=h->values[(h->next+QR_WINDOW-1)%QR_WINDOW][9];
      qr_rtk_destroy(ctx);free(r);return value;
    }
    '''
    p=tmp_path/'e.c';p.write_text(stub);out=tmp_path/'e.so'
    subprocess.run(['cc','-std=c99','-shared','-fPIC','-I'+str(KIT/'src'),str(p),'-lm','-o',str(out)],check=True)
    lib=ct.CDLL(str(out));lib.evidence_check.argtypes=[ct.c_int];lib.evidence_check.restype=ct.c_double
    assert abs(lib.evidence_check(0))>.1
    assert abs(lib.evidence_check(1))<1e-12


def test_qr_candidate_gradient_matches_native_finite_difference(tmp_path):
    """Two fixed-path EKF updates + given-integer conditioning, not search AD."""
    import subprocess,math
    from qrlearn.scl import condition_position
    stub='#include "rtklib.h"\n#include "qr_filter.h"\nint step(double*x,double*P,const double*H,const double*v,const double*R,int n,int m){return qr_stable_filter(x,P,H,v,R,n,m);}\n'
    p=tmp_path/'f.c';p.write_text(stub);so=tmp_path/'f.so'
    subprocess.run(['cc','-O2','-std=c99','-shared','-fPIC','-I'+str(ROOT/'src'),str(p),'-L'+str(ROOT/'lib'),'-lrtklib','-lm','-Wl,-rpath,'+str(ROOT/'lib'),'-o',str(so)],check=True)
    lib=ct.CDLL(str(so));dp=ct.POINTER(ct.c_double);lib.step.argtypes=[dp]*5+[ct.c_int]*2
    rng=np.random.default_rng(52);n=12;m=6;B=rng.normal(size=(n,n));P0=B@B.T*.1+np.eye(n)*.2;x0=np.arange(n,dtype=float)+1.3
    F=np.eye(n);F[np.arange(6),np.arange(6)+3]=.5;F[np.arange(3),np.arange(3)+6]=.125
    Q=np.diag([0]*6+[.08]*3+[0]*3);H=rng.normal(size=(m,n))*.2;G=rng.normal(size=(m,m));R=G@G.T*.1+np.eye(m)*.4
    z=H@x0+rng.normal(size=m);pairs=torch.tensor([[9,10],[9,11]]);integer=torch.tensor([-1.,-2.],dtype=torch.float64)
    def native(q,r):
        x=x0.copy();P=P0.copy()
        for _ in range(2):
            x=np.ascontiguousarray(F@x);P=np.asfortranarray(F@P@F.T+q*Q);hh=np.asfortranarray(H.T);v=np.ascontiguousarray(z-H@x);rr=np.asfortranarray(r*R)
            assert lib.step(*(a.ctypes.data_as(dp) for a in [x,P,hh,v,rr]),n,m)==0
        D=np.zeros((2,n));D[:,9]=1;D[0,10]=-1;D[1,11]=-1
        y=x[:3]-P[:3]@D.T@np.linalg.solve(D@P@D.T,D@x-integer.numpy())
        return (y*y).sum()+.1*(x[:3]**2).sum()
    q=torch.tensor(1.2,dtype=torch.float64,requires_grad=True);r=torch.tensor(1.8,dtype=torch.float64,requires_grad=True)
    t=lambda x:torch.tensor(x,dtype=torch.float64);x=t(x0);P=t(P0);Ft,Ht,Qt,Rt,zt=map(t,[F,H,Q,R,z])
    for _ in range(2):
        x=Ft@x;P=Ft@P@Ft.T+q*Qt;S=Ht@P@Ht.T+r*Rt;K=torch.linalg.solve(S,Ht@P).T
        x=x+K@(zt-Ht@x);A=torch.eye(n)-K@Ht;P=A@P@A.T+K@(r*Rt)@K.T
    y=condition_position(x,P,pairs,integer);loss=y.square().sum()+.1*x[:3].square().sum();loss.backward()
    assert loss.item()==pytest.approx(native(1.2,1.8),rel=1e-10)
    h=1e-5
    numeric=[(native(1.2+h,1.8)-native(1.2-h,1.8))/(2*h),(native(1.2,1.8+h)-native(1.2,1.8-h))/(2*h)]
    np.testing.assert_allclose([q.grad.item(),r.grad.item()],numeric,atol=1e-7,rtol=2e-5)

@pytest.mark.parametrize('flag,value,match',[
 ('--conditional-weight','-1','conditional_weight'),('--smooth-weight','nan','smooth_weight'),
 ('--gradient-stride','0','gradient_stride')])
def test_invalid_scl_training_controls_rejected_before_file_access(flag,value,match,tmp_path):
    from qrlearn.cli import parser
    from qrlearn.training import train
    args=parser().parse_args(['train','--manifest',str(tmp_path/'missing.json'),'--config',str(tmp_path/'missing.conf'),'--output',str(tmp_path/'out'),flag,value])
    with pytest.raises(ValueError,match=match):train(args)


def test_v3_rejects_claimed_phase_adaptation(tmp_path):
    """V3 fixes phase weights; a header must not claim a discarded phase policy."""
    import subprocess
    m=QRModel(r_policy='scl').double();p=tmp_path/'bad.qr'
    m.export(p,provenance='synthetic')
    text=p.read_text();lines=text.splitlines();fields=lines[1].split();fields[-1]='4';lines[1]=' '.join(fields);p.write_text('\n'.join(lines)+'\n')
    c=tmp_path/'load.c';c.write_text('#include "learned_qr.h"\nint main(int n,char**v){qr_model m;return qr_load(&m,v[1],1)?0:3;}\n')
    exe=tmp_path/'load';subprocess.run(['cc','-std=c99','-I'+str(KIT/'src'),str(c),'-lm','-o',str(exe)],check=True)
    assert subprocess.run([str(exe),str(p)]).returncode==3


def test_supervision_includes_native_dgps_filter_updates_not_single_point():
    from qrlearn import training
    assert hasattr(training,'has_filter_state'),'native DGPS-filter state supervision is missing'
    assert training.has_filter_state(2) and training.has_filter_state(4)
    assert not training.has_filter_state(0) and not training.has_filter_state(5)
    assert not training.has_filter_state(1),'training remains continuous, not a FIX state graph'
