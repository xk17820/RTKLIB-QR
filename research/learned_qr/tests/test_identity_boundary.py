"""Boundary contracts, including an explicit zero-subgradient fault injection.

The injection is NOT a reproduction of a particular installed Torch version. It
preserves clamp's forward values and changes only its undefined derivative at
zero, to prove our inflation policy does not inherit that backend convention.
"""
import math
import pytest
import torch
from qrlearn.model import QRModel


def history(dtype=torch.float64):
    h = torch.zeros(10, 13, dtype=dtype)
    h[:, -1] = 1
    h[:, 0] = -1.0  # code_guard: 30 dB-Hz
    h[:, 1] = 0.5
    h[:, 3] = 1.0
    h[:, 8] = math.log1p(12.0)  # scl: causal anomaly evidence
    return h


@pytest.mark.parametrize('policy', ['code_guard', 'scl', 'scl_blind'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_identity_gradient_does_not_inherit_clamp_boundary(policy, dtype, monkeypatch):
    original = torch.Tensor.clamp

    def zero_boundary(self, *args, **kwargs):
        if not args and kwargs == {'min': 0.0}:
            # Same forward function, a different admissible boundary derivative.
            return torch.where(self > 0, self, torch.zeros_like(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'clamp', zero_boundary)
    m = QRModel(r_policy=policy).to(dtype=dtype)
    value = m.r(history(dtype))
    assert value.item() == 1.0
    value.backward()
    assert m.r.fc2.bias.grad.item() == pytest.approx(math.log(m.r.limit), rel=2e-6)


@pytest.mark.parametrize('policy', ['code_guard', 'scl', 'scl_blind'])
def test_identity_uses_right_derivative_and_can_take_first_step(policy):
    m = QRModel(r_policy=policy).double()
    h = history()
    initial = m.r(h)
    initial.backward()
    analytic = m.r.fc2.bias.grad.item()
    eps = 1e-7
    with torch.no_grad():
        m.r.fc2.bias.fill_(eps)
        numerical = (m.r(h).item() - 1.0) / eps
        m.r.fc2.bias.zero_()
    assert analytic == pytest.approx(numerical, rel=1e-6)
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-3)
    optimizer.zero_grad()
    (m.r(h) - 2.0).square().sum().backward()
    optimizer.step()
    assert 1.0 < m.r(h).item() <= m.r.limit


@pytest.mark.parametrize('policy', ['code_guard', 'scl', 'scl_blind'])
def test_negative_head_phase_and_missing_history_remain_unlearned_identity(policy):
    m = QRModel(r_policy=policy).double()
    with torch.no_grad():
        m.r.fc2.bias.fill_(-0.4)
    cases = [history()]
    phase = history(); phase[:, 3] = 0.0
    missing = history(); missing[:, -1] = 0.0
    for h in cases:
        m.zero_grad(); value = m.r(h); value.backward()
        assert value.item() == 1.0
        assert m.r.fc2.bias.grad.item() == 0.0
    with torch.no_grad():
        m.r.fc2.bias.fill_(0.4)
    for h in [phase, missing]:
        m.zero_grad(); value = m.r(h); value.backward()
        assert value.item() == 1.0
        assert m.r.fc2.bias.grad.item() == 0.0
