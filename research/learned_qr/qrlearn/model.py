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

class QRModel(nn.Module):
    def __init__(self, q_limit: float = 10.0, r_limit: float = 1000.0):
        super().__init__()
        if not 1.0 < q_limit <= 100.0:
            raise ValueError("q_limit must match native safety cap: (1,100]")
        self.q = ScaleNet(Q_FEATURES, 2, q_limit)
        self.r = ScaleNet(R_FEATURES, 1, r_limit)

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
        parts = [f'RTKLIB_QR_V1 {codes[provenance]} {self.q.limit:.17g} {self.r.limit:.17g} {max_gap:.17g}\n']
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
