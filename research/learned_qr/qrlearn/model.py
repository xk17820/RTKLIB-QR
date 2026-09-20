"""Temporal MLPs producing bounded *variance* multipliers, not coordinates."""
from __future__ import annotations
from pathlib import Path
import math
import torch
from torch import nn

WINDOW, Q_FEATURES, R_FEATURES, HIDDEN = 10, 10, 12, 16

class ScaleNet(nn.Module):
    def __init__(self, features: int, outputs: int, limit: float):
        super().__init__()
        if not 1.0 < limit <= 1e6:
            raise ValueError('limit must be in (1, 1e6]')
        self.features, self.outputs, self.limit = features, outputs, limit
        self.fc1 = nn.Linear(WINDOW * (features + 1), HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, outputs)
        # Exactly reproduces baseline scaling until training changes the head.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-2:] != (WINDOW, self.features + 1):
            raise ValueError(f'expected (...,{WINDOW},{self.features+1}) history')
        if not torch.isfinite(history).all():
            raise ValueError('history contains non-finite values')
        x = history.flatten(start_dim=-2)
        return torch.exp(math.log(self.limit) * torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))

def _nonnegative_right(value: torch.Tensor) -> torch.Tensor:
    """Positive part with an explicit RIGHT derivative of one at zero.

    The forward value is exactly max(value, 0), as in native C. The kink has no
    two-sided derivative: choosing its right derivative lets an identity head
    start learning inflation. A negative head still has zero gradient; this is
    not a straight-through estimator outside the boundary. Do not replace with
    ReLU/clamp and implicitly inherit a backend's equality convention.
    """
    return torch.where(value >= 0.0, value, torch.zeros_like(value))


class GuardedCodeNet(ScaleNet):
    """Learned code inflation with explicit quality constraints.

    The phase guard is a fixed, bounded policy, NOT a learned phase/bias head.
    C/N0 is only a heuristic; high-C/N0 NLOS can escape this gate. Missing C/N0
    (zero) is not interpreted as an unreliable signal. Validate receiver-specific
    thresholds on independent routes before deployment.
    """
    def __init__(self,limit,anchor=40.,width=10.,phase_cap=1.):
        if not math.isfinite(anchor) or not 20<=anchor<=55:
            raise ValueError('guard anchor must be in [20,55] dB-Hz')
        if not math.isfinite(width) or not 1<=width<=30:
            raise ValueError('guard width must be in [1,30] dB-Hz')
        if not math.isfinite(phase_cap) or not 1<=phase_cap<=100:
            raise ValueError('phase cap must be in [1,100]')
        super().__init__(R_FEATURES,1,limit)
        self.anchor,self.width,self.phase_cap=anchor,width,phase_cap

    def forward(self,history):
        raw=super().forward(history);last=history[...,-1,:]
        cn0=40.+10.*last[...,:2]
        deficit=torch.where(cn0>0.,(self.anchor-cn0)/self.width,torch.zeros_like(cn0))
        gate=deficit.amax(-1).clamp(0.,1.)*(last[...,-1]>.5)
        code=1.+gate.unsqueeze(-1)*_nonnegative_right(raw-1.)
        phase=1.+gate.unsqueeze(-1)*(self.phase_cap-1.)
        return torch.where((last[...,3]>.5).unsqueeze(-1),code,phase)

class SCLCodeNet(ScaleNet):
    """Causal consistency anchored code covariance; carrier-phase is unchanged.

    f8=clock-centered standardized code innovation, f9=dual-frequency MP
    deviation from a causal anchor. Values are signed-log compressed in C.
    C/N0 is a feature, never the gate. Missing evidence implies identity.
    """
    def __init__(self,limit,threshold=2.,width=4.,use_consistency=True,inflation_only=True):
        if not (0<threshold<=55 and 1<=width<=30):raise ValueError('invalid evidence gate')
        super().__init__(R_FEATURES,1,limit)
        self.anchor,self.width,self.phase_cap=threshold,width,1.
        self.use_consistency=use_consistency
        self.inflation_only=inflation_only
    def forward(self,history):
        h=history
        if not self.use_consistency:
            h=history.clone();h[...,8:11]=0.
        raw=super().forward(h);last=history[...,-1,:]
        evidence=torch.expm1(last[...,8:10].abs().clamp(max=8)).amax(-1)
        gate=((evidence-self.anchor)/self.width).clamp(0.,1.) if self.use_consistency else torch.ones_like(evidence)
        gate=gate*(last[...,3]>.5)*(last[...,-1]>.5)
        return 1.+gate.unsqueeze(-1)*(_nonnegative_right(raw-1.) if self.inflation_only else raw-1.)

class QRModel(nn.Module):
    def __init__(self, q_limit: float = 10.0, r_limit: float = 1000.0, *,
                 r_policy='unconstrained',guard_anchor=40.,guard_width=10.,phase_guard_cap=1.,
                 evidence_threshold=2.,evidence_width=4.):
        super().__init__()
        if not 1.0 < q_limit <= 100.0:
            raise ValueError("q_limit must match native safety cap: (1,100]")
        self.q = ScaleNet(Q_FEATURES, 2, q_limit)
        if r_policy not in ('unconstrained','code_guard','scl','scl_blind','scl_unconstrained'):raise ValueError('unknown R policy')
        self.r_policy=r_policy
        self.r = (GuardedCodeNet(r_limit,guard_anchor,guard_width,phase_guard_cap)
                  if r_policy=='code_guard' else ScaleNet(R_FEATURES,1,r_limit))
        if r_policy in ('scl','scl_blind','scl_unconstrained'):
            self.r=SCLCodeNet(r_limit,evidence_threshold,evidence_width,r_policy!='scl_blind',r_policy!='scl_unconstrained')

    def forward(self, q_history: torch.Tensor, r_history: torch.Tensor):
        return self.q(q_history), self.r(r_history)

    def export(self, path: str | Path, *, provenance: str, max_gap: float = 30.0):
        """Export finite numerical weights only. No pickle/ONNX/runtime dependency.

        `trained` records the caller's provenance declaration, not an accuracy certification.
        Native deployment refuses `untrained`/`synthetic` unless explicitly opted in.
        """
        codes = {'untrained': 0, 'trained': 1, 'synthetic': 2}
        if provenance not in codes or not math.isfinite(max_gap) or not 0 < max_gap <= 3600:
            raise ValueError('invalid provenance or max_gap')
        magic='RTKLIB_QR_V3' if self.r_policy in ('scl','scl_blind','scl_unconstrained') else 'RTKLIB_QR_V2' if self.r_policy=='code_guard' else 'RTKLIB_QR_V1'
        parts=[f'{magic} {codes[provenance]} {self.q.limit:.17g} {self.r.limit:.17g} {max_gap:.17g}\n']
        if self.r_policy!='unconstrained':
            policy={'code_guard':1,'scl':2,'scl_blind':3,'scl_unconstrained':4}[self.r_policy]
            parts.append(f'{policy} {self.r.anchor:.17g} {self.r.width:.17g} {self.r.phase_cap:.17g}\n')
        for net in (self.q, self.r):
            parts.append(f'{WINDOW*(net.features+1)} {net.outputs}\n')
            for parameter in (net.fc1.weight, net.fc1.bias, net.fc2.weight, net.fc2.bias):
                a = parameter.detach().cpu().double().reshape(-1)
                if not torch.isfinite(a).all() or (a.abs() > 1e6).any():
                    raise ValueError('weights must be finite and have magnitude <= 1e6')
                parts.append(' '.join(format(v, '.17g') for v in a.tolist()) + '\n')
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        temporary = p.with_name(p.name + '.tmp')
        temporary.write_text(''.join(parts), encoding='ascii')
        temporary.replace(p)
