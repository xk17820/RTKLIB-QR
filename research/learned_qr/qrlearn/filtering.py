"""Differentiable covariance/Kalman primitives; no integer search is differentiated.

H uses conventional (measurement,state) shape; RTKLIB's storage is transposed.
All inputs must already be in consistent units (including metre-valued phase residuals).
"""
from __future__ import annotations
from collections.abc import Callable
import torch

def _finite(name: str, value: torch.Tensor):
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains non-finite values')

def dd_covariance(D: torch.Tensor, sd_variance: torch.Tensor) -> torch.Tensor:
    if D.ndim != 2 or sd_variance.ndim != 1 or D.shape[1] != sd_variance.numel() or D.shape[0] == 0:
        raise ValueError('D must be a nonempty (DD,SD) matrix matching the SD variance vector')
    _finite('D', D); _finite('sd_variance', sd_variance)
    if (sd_variance <= 0).any():
        raise ValueError('SD variances must be positive')
    return (D * sd_variance.unsqueeze(0)) @ D.mT

def predict(x: torch.Tensor, P: torch.Tensor, F: torch.Tensor, Q: torch.Tensor):
    for name, t in (('x',x),('P',P),('F',F),('Q',Q)):
        _finite(name,t)
    cov = F @ P @ F.mT + Q
    return F @ x, 0.5 * (cov + cov.mT)

def ekf_update(x: torch.Tensor, P: torch.Tensor, z: torch.Tensor,
               observe: Callable[[torch.Tensor], torch.Tensor], H: torch.Tensor,
               R: torch.Tensor):
    """Return posterior x,P, normalized innovation square, and Gaussian NLL.

    `observe(x)` is called on the CURRENT predicted state on every invocation.
    A GNSS adapter must recompute its Jacobian at that same state; archived baseline
    residuals are not a valid replacement. Discrete gating/AR are outside this graph.
    Singular/non-positive innovation covariance raises instead of hiding the failure.
    """
    if x.ndim != 1 or z.ndim != 1 or z.numel() == 0:
        raise ValueError('nonempty one-dimensional state and observations required')
    if H.shape != (z.numel(), x.numel()) or R.shape != (z.numel(), z.numel()) or P.shape != (x.numel(), x.numel()):
        raise ValueError('inconsistent filter dimensions')
    for name,t in (('x',x),('P',P),('z',z),('H',H),('R',R)):
        _finite(name,t)
    residual = z - observe(x)
    _finite('residual',residual)
    S = H @ P @ H.mT + R
    S = 0.5 * (S + S.mT)
    L = torch.linalg.cholesky(S)
    K = torch.cholesky_solve((P @ H.mT).mT, L).mT
    posterior = x + K @ residual
    A = torch.eye(x.numel(),dtype=x.dtype,device=x.device) - K @ H
    cov = A @ P @ A.mT + K @ R @ K.mT
    cov = 0.5 * (cov + cov.mT)
    nis = residual @ torch.cholesky_solve(residual.unsqueeze(-1),L).squeeze(-1)
    nll = 0.5 * (nis + 2.0*torch.log(torch.diagonal(L)).sum() + z.numel()*torch.log(x.new_tensor(2*torch.pi)))
    return posterior,cov,nis,nll
