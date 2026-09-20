"""The identity R-head has an explicit right derivative, not a clamp convention."""
import math

import pytest
import torch

from qrlearn.model import QRModel


def history(policy, gate, dtype):
    h = torch.zeros((10, 13), dtype=dtype)
    h[:, -1] = 1
    h[:, 3] = 1  # code, not carrier phase
    h[:, 0] = -gate  # code_guard: rover C/N0 = 40 - 10 * gate
    h[:, 1] = .5
    h[:, 8] = math.log1p(2 + 4 * gate)  # SCL evidence gate
    return h


def primitive_probe(dtype):
    x = torch.tensor(0., dtype=dtype, requires_grad=True)
    y = x.clamp(min=0.)
    a, = torch.autograd.grad(y, x)
    z = torch.where(x >= 0., x, torch.zeros_like(x))
    b, = torch.autograd.grad(z, x)
    return {'torch': torch.__version__, 'clamp_zero': a.item(),
            'where_ge_zero': b.item()}


@pytest.mark.parametrize('policy', ['code_guard', 'scl', 'scl_blind'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('gate', [.5, 1.])
def test_identity_head_uses_right_derivative(policy, dtype, gate):
    model = QRModel(r_policy=policy).to(dtype=dtype)
    h = history(policy, gate, dtype)
    out = model.r(h)
    assert out.item() == 1.
    out.backward()
    effective_gate = 1. if policy == 'scl_blind' else gate
    expected = effective_gate * math.log(model.r.limit)
    assert model.r.fc2.bias.grad.item() == pytest.approx(expected, rel=2e-6), primitive_probe(dtype)


@pytest.mark.parametrize('policy', ['code_guard', 'scl', 'scl_blind'])
def test_one_optimizer_step_can_leave_identity(policy):
    model = QRModel(r_policy=policy).double()
    h = history(policy, 1., torch.float64)
    opt = torch.optim.SGD(model.r.parameters(), lr=1e-3)
    loss = (model.r(h) - 2.).square().sum()
    loss.backward()
    opt.step()
    assert 1. < model.r(h).item() < model.r.limit, primitive_probe(torch.float64)


@pytest.mark.parametrize('policy', ['code_guard', 'scl', 'scl_blind'])
def test_lower_branch_remains_identity_and_has_zero_gradient(policy):
    model = QRModel(r_policy=policy).double()
    with torch.no_grad():
        model.r.fc2.bias.fill_(-.2)
    out = model.r(history(policy, 1., torch.float64))
    out.backward()
    assert out.item() == 1.
    assert model.r.fc2.bias.grad.item() == 0.
