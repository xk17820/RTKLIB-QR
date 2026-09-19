from pathlib import Path
import importlib
import numpy as np
import pytest
import torch
KIT=Path(__file__).resolve().parents[1]

def module():
    assert (KIT/'qrlearn/reference.py').exists(), 'reference reader/loss not implemented'
    return importlib.import_module('qrlearn.reference')

def write(tmp_path,rows):
    p=tmp_path/'reference.csv'
    p.write_text('gps_week,tow_s,x_m,y_m,z_m,valid\n'+rows)
    return p

def test_interpolation_validity_gaps_and_no_extrapolation(tmp_path):
    m=module();p=write(tmp_path,'2400,0,6378137,0,0,1\n2400,1,6378137,2,0,1\n2400,2,nan,nan,nan,0\n2400,10,6378137,20,0,1\n')
    r=m.Reference.load(p,max_gap=2)
    x,ok=r.at(2400*604800+.5);assert ok;np.testing.assert_allclose(x,[6378137,1,0])
    assert not r.at(2400*604800-1)[1]
    assert not r.at(2400*604800+1.5)[1]
    assert not r.at(2400*604800+5)[1]
    assert not r.at(2400*604800+11)[1]

def test_duplicate_time_and_wrong_units_rejected(tmp_path):
    m=module();p=write(tmp_path,'2400,0,6378137,0,0,1\n2400,0,6378137,0,0,1\n')
    with pytest.raises(ValueError,match='increasing'):m.Reference.load(p)
    p=write(tmp_path,'2400,0,35,120,20,1\n')
    with pytest.raises(ValueError,match='ECEF'):m.Reference.load(p)

def test_enu_loss_uses_reference_only_as_target_and_has_gradients():
    m=module();ref=np.array([6378137.,0.,0.]);p=torch.tensor(ref+[3.,1.,2.],dtype=torch.float64,requires_grad=True)
    error=m.enu_error(p,ref)
    np.testing.assert_allclose(error.detach(),[1,2,3],atol=1e-12)
    loss=m.position_loss(p,ref,up_weight=2,huber_delta=0)
    loss.backward();np.testing.assert_allclose(p.grad,[12,2,4],atol=1e-12)

def test_no_reference_in_native_api():
    import inspect
    from qrlearn.native import Session
    assert 'reference' not in inspect.signature(Session).parameters

def test_evaluation_reports_coverage_and_fixed_error_proxy():
    m=module()
    metrics=m.summarize(np.array([[1.,0,0],[0,0,2.]]),np.array([1,2]),attempted=4,fix_threshold=.5)
    assert metrics['attempted_epochs']==4 and metrics['matched_epochs']==2
    assert metrics['fixed_large_error_proxy_count']==1
    assert metrics['rms_3d_m']==pytest.approx(np.sqrt(2.5))
