import math
import numpy as np
import pytest
from qrlearn.rules import variance_multiplier,RuleController
from qrlearn.native import Session
from tests_support import raw_kwargs


def history(code=1,valid=1,innovation=0,mp=0):
    h=np.zeros((10,13));h[-1,3]=code;h[-1,-1]=valid
    h[-1,8]=np.log1p(innovation);h[-1,9]=np.log1p(mp)
    return h


@pytest.mark.parametrize('method',['fixed','huber'])
def test_clean_phase_missing_observations_remain_identity(method):
    for h in [history(),history(code=0,innovation=20),history(valid=0,innovation=20)]:
        assert variance_multiplier(h,method,gain=100,k=1.5)==1.


def test_fixed_rule_uses_same_scl_evidence_gate():
    assert variance_multiplier(history(innovation=2),'fixed',gain=101)==1.
    assert variance_multiplier(history(innovation=4),'fixed',gain=101)==pytest.approx(51.)
    assert variance_multiplier(history(mp=8),'fixed',gain=101)==101.


def test_huber_is_inverse_irls_weight_not_squared_unless_explicitly_defined():
    assert variance_multiplier(history(innovation=6),'huber',k=1.5)==pytest.approx(4.)
    assert variance_multiplier(history(innovation=10000),'huber',k=1.5)<=1000.


def test_rule_identity_keeps_full_native_ar_output_and_repeated_runs(tmp_path):
    opts=raw_kwargs()
    rows=[]
    for live in (False,True):
        out=[];ctrl=RuleController('fixed',gain=1.)
        with Session(**opts,mode='r' if live else 'off',training=False,scl=live) as s:
            while (r:=s.step(ctrl.handle if live else None)) is not None:out.append(r)
        rows.append(out)
    assert len(rows[0])==len(rows[1])==120
    for a,b in zip(*rows):
        assert a['time']==b['time'] and a['status']==b['status'] and a['ratio']==b['ratio']
        np.testing.assert_array_equal(a['position'],b['position'])
