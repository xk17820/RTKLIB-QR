import importlib.util
import numpy as np
import pytest
import torch

def mod(name):
    assert importlib.util.find_spec(name) is not None, f'{name} must be implemented'
    return __import__(name, fromlist=['*'])

def test_identity_initialization_and_bounds():
    m=mod('qrlearn.model'); net=m.QRModel().double()
    q,r=net(torch.randn(3,10,11,dtype=torch.float64),torch.randn(5,10,13,dtype=torch.float64))
    torch.testing.assert_close(q,torch.ones_like(q)); torch.testing.assert_close(r,torch.ones_like(r))
    with torch.no_grad(): net.q.fc2.bias.fill_(1e9); net.r.fc2.bias.fill_(-1e9)
    q,r=net(torch.zeros(1,10,11,dtype=torch.float64),torch.zeros(1,10,13,dtype=torch.float64))
    assert (q<=10.00000001).all() and (r>=0.0009999999).all()

def test_covariance_preserves_common_reference_and_gradients():
    f=mod('qrlearn.filtering')
    sd=torch.tensor([4.,9.,16.],dtype=torch.float64,requires_grad=True)
    d=torch.tensor([[1.,-1.,0.],[1.,0.,-1.]],dtype=torch.float64)
    R=f.dd_covariance(d,sd)
    torch.testing.assert_close(R,torch.tensor([[13.,4.],[4.,20.]],dtype=torch.float64))
    assert (torch.linalg.eigvalsh(R)>0).all()
    R.sum().backward(); torch.testing.assert_close(sd.grad,torch.tensor([4.,1.,1.],dtype=torch.float64))

def test_joseph_update_matches_scalar_solution():
    f=mod('qrlearn.filtering'); d=torch.float64
    x,P,nis,nll=f.ekf_update(torch.tensor([0.],dtype=d),torch.tensor([[4.]],dtype=d),
        torch.tensor([3.],dtype=d),lambda x:x,torch.ones(1,1,dtype=d),torch.tensor([[5.]],dtype=d))
    torch.testing.assert_close(x,torch.tensor([4./3],dtype=d))
    torch.testing.assert_close(P,torch.tensor([[20./9]],dtype=d))
    assert nis.item()==pytest.approx(1.) and torch.isfinite(nll)

def test_nonlinear_residual_is_recomputed_at_current_state():
    f=mod('qrlearn.filtering'); d=torch.float64
    x=torch.tensor([2.],dtype=d,requires_grad=True)
    out=f.ekf_update(x,torch.eye(1,dtype=d),torch.tensor([9.],dtype=d),
        lambda t:t.square(),torch.tensor([[4.]],dtype=d),torch.eye(1,dtype=d))
    assert out[0].item()==pytest.approx(2+4/17*5)
    out[0].sum().backward(); assert torch.isfinite(x.grad).all()

def test_covariance_rejects_negative_or_nan_variance():
    f=mod('qrlearn.filtering')
    for value in (-1.,float('nan')):
        with pytest.raises(ValueError): f.dd_covariance(torch.eye(2),torch.tensor([1.,value]))

def test_all_masked_and_bad_inputs_are_not_silently_accepted():
    f=mod('qrlearn.filtering')
    with pytest.raises(ValueError): f.dd_covariance(torch.empty(0,3),torch.ones(3))

def test_backward_through_two_predictions_updates_both_networks():
    m=mod('qrlearn.model'); f=mod('qrlearn.filtering'); d=torch.float64
    net=m.QRModel().double(); x=torch.zeros(2,dtype=d); P=torch.eye(2,dtype=d)
    F=torch.tensor([[1.,1.],[0.,1.]],dtype=d); H=torch.tensor([[1.,0.]],dtype=d)
    total=0
    for k in range(3):
        q,r=net(torch.ones(1,10,11,dtype=d),torch.ones(1,10,13,dtype=d))
        x,P=f.predict(x,P,F,torch.diag(q[0]*torch.tensor([.2,.1],dtype=d)))
        x,P,nis,nll=f.ekf_update(x,P,torch.tensor([1.+k],dtype=d),lambda t:H@t,H,r.reshape(1,1))
        total=total+(x[0]-(.5+k))**2+.01*nll
    total.backward()
    assert net.q.fc2.bias.grad.abs().sum()>0
    assert net.r.fc2.bias.grad.abs().sum()>0

def test_q_limit_matches_native_loader_safety_cap():
    m=mod('qrlearn.model')
    with pytest.raises(ValueError):m.QRModel(q_limit=101)

def test_export_rejects_weights_outside_native_numeric_contract(tmp_path):
    m=mod('qrlearn.model');net=m.QRModel().double()
    with torch.no_grad():net.q.fc2.bias.fill_(1e7)
    with pytest.raises(ValueError):net.export(tmp_path/'invalid.qr',provenance='synthetic')
