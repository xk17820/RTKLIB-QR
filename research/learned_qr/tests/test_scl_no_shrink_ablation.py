"""One-factor ablation: identical V3 evidence, gate, phase and DD; allow code shrink."""
import math
import numpy as np
import pytest
import torch
from qrlearn.model import QRModel
from qrlearn.training import checkpoint_save,checkpoint_load
from test_export_boundary_parity import evaluator


@pytest.mark.parametrize('head',[-.4,0.,.4])
def test_same_feature_gate_native_export_and_only_negative_side_changes(tmp_path,evaluator,head):
    base=QRModel(r_policy='scl').double(); free=QRModel(r_policy='scl_unconstrained').double()
    free.load_state_dict(base.state_dict())
    with torch.no_grad():
        base.r.fc2.bias.fill_(head);free.r.fc2.bias.fill_(head)
    path=tmp_path/'free.qr';free.export(path,provenance='synthetic')
    assert path.read_text().splitlines()[0].startswith('RTKLIB_QR_V3')
    assert path.read_text().splitlines()[1].startswith('4 ')
    lib,D=evaluator
    for evidence in [0.,3.,12.]:
        for code in [0.,1.]:
            h=torch.zeros(10,13,dtype=torch.float64);h[:,-1]=1;h[:,3]=code;h[:,8]=math.log1p(evidence)
            v=free.r(h).item();b=base.r(h).item();a=h.numpy();out=np.zeros(1)
            assert lib.evaluate(str(path).encode(),a.ctypes.data_as(D),out.ctypes.data_as(D))==1
            assert out[0]==pytest.approx(v,rel=3e-12,abs=3e-12)
            if evidence<=2 or code==0:assert v==b==1.
            elif head>=0:assert v==b
            else:assert 0.<v<1. and b==1.
    checkpoint_save(tmp_path/'checkpoint.pt',free,max_gap=30.,epoch=1,mode='qr')
    loaded,_=checkpoint_load(tmp_path/'checkpoint.pt')
    assert loaded.r_policy=='scl_unconstrained'
    assert loaded.r(h).item()==free.r(h).item()


def test_unconstrained_side_retains_actual_gradient_below_identity():
    m=QRModel(r_policy='scl_unconstrained').double()
    with torch.no_grad():m.r.fc2.bias.fill_(-.1)
    h=torch.zeros(10,13,dtype=torch.float64);h[:,-1]=1;h[:,3]=1;h[:,8]=math.log1p(12)
    m.r(h).sum().backward()
    assert m.r.fc2.bias.grad.abs().sum()>0
