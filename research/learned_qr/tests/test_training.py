import importlib.util
import math
import pytest
import torch

def load(name):
    assert importlib.util.find_spec(name) is not None,f'{name} must be implemented'
    return __import__(name,fromlist=['*'])

def test_feature_contract_values_and_differentiable_state_features():
    f=load('qrlearn.features');d=torch.float64
    velocity=torch.tensor([30.,-15.,0.],dtype=d,requires_grad=True)
    q=f.q_features(1.,velocity,torch.tensor([3.,0.,-3.],dtype=d),torch.tensor(4.,dtype=d),20,2)
    torch.testing.assert_close(q,torch.tensor([math.log(2),1.,-.5,0.,1.,0.,-1.,math.log(3),.5,1./3],dtype=d))
    q.sum().backward();assert velocity.grad.abs().sum()>0
    r=f.r_features(42.,43.,.7,False,0,3,0,1000.,0.,0.,False,None,1)
    assert r[0].item()==pytest.approx(.2) and r[2].item()==pytest.approx(math.sin(.7))
    assert r[-1].item()==pytest.approx(math.log(2))

def test_temporal_window_causal_order_and_reset():
    f=load('qrlearn.features');w=f.TemporalWindow(10)
    x=torch.arange(10,dtype=torch.float64,requires_grad=True)
    h=w.push(1,3,x);assert h.shape==(10,11) and h[:-1].sum()==0 and h[-1,-1]==1
    torch.testing.assert_close(w.push(1,3,x+1,reset=True),h)
    assert w.push(2,3,x+1)[-2:,-1].sum()==2
    assert w.push(50,3,x)[...,-1].sum()==1
    w.push(51,3,x)[-1,:10].sum().backward();assert (x.grad==1).all()

def test_synthetic_smoke_updates_both_heads_and_marks_provenance(tmp_path):
    m=load('examples.train_synthetic')
    p=tmp_path/'toy.qr';report=m.train(epochs=4,output=p,length=12)
    assert report['data_kind']=='synthetic_non_GNSS'
    assert report['q_head_changed'] and report['r_head_changed']
    assert math.isfinite(report['initial_training_loss']) and math.isfinite(report['final_training_loss'])
    assert p.read_text().startswith('RTKLIB_QR_V1 2 ')
