"""Feature schema V1 and causal temporal windows matching the C runtime.

No reference position, future observation, RTK error label or learned bias is an
input feature. A real GNSS trainer must evaluate these on its current live state.
"""
from __future__ import annotations
import math
import torch
from .model import WINDOW

def q_features(dt: float, velocity_enu: torch.Tensor, acceleration_enu: torch.Tensor,
               mean_position_variance: torch.Tensor, ns: int, status: int) -> torch.Tensor:
    if velocity_enu.shape!=(3,) or acceleration_enu.shape!=(3,) or dt<=0 or not math.isfinite(dt):
        raise ValueError('positive forward dt and three-axis ENU vectors required')
    var=mean_position_variance.to(velocity_enu)
    if not torch.isfinite(var):raise ValueError('non-finite position variance')
    std=torch.where(var>0,torch.sqrt(var.clamp_min(1e-24)),torch.zeros_like(var))
    return torch.cat((velocity_enu.new_tensor([math.log1p(dt)]),velocity_enu/30,
                      acceleration_enu/3,torch.log1p(std).reshape(1),
                      velocity_enu.new_tensor([ns/40,status/6])))

def r_features(snr_rover: float,snr_base: float,elevation: float,code: bool,
               frequency_slot: int,nfreq_compiled: int,system_index: int,
               baseline_m: float,age_s: float,reported_std: float,half_cycle: bool,
               previous_normalized_cn0: float | None,tracked_count: int) -> torch.Tensor:
    """Reported std is exactly obs.Pstd or obs.Lstd as exposed by the receiver parser.
    frequency_slot is zero-based. system_index: GPS,GLO,GAL,BDS,QZS,SBAS,IRN=0..6.
    previous_normalized_cn0 is None after a gap/slip/signal change, NOT zero-filled.
    """
    if not 0<=system_index<=6 or not 0<=frequency_slot<nfreq_compiled or tracked_count<1:
        raise ValueError('invalid system/frequency/tracking information')
    t=lambda x:torch.as_tensor(x,dtype=torch.float64)
    slog=lambda x:torch.sign(t(x))*torch.log1p(torch.abs(t(x)))
    cn0=(t(snr_rover)-40)/10
    delta=cn0-t(previous_normalized_cn0) if previous_normalized_cn0 is not None else t(0)
    return torch.stack((cn0,(t(snr_base)-40)/10,torch.sin(t(elevation)),t(code),
        t(frequency_slot/max(nfreq_compiled-1,1)),t(system_index/6),
        torch.log1p(torch.abs(t(baseline_m))/1000),slog(age_s),slog(reported_std),
        t(half_cycle),delta,torch.log1p(t(tracked_count))))

class TemporalWindow:
    def __init__(self,features: int,max_gap: float=30.0):
        if features<1 or max_gap<=0:raise ValueError('invalid window configuration')
        self.features,self.max_gap=features,max_gap
        self.frames:list[torch.Tensor]=[]
        self.last_time:float|None=None
        self.signature:int|None=None
        self.seen=0

    def push(self,time: float,signature: int,feature: torch.Tensor,*,reset: bool=False):
        if feature.shape!=(self.features,) or not torch.isfinite(feature).all() or not math.isfinite(time):
            raise ValueError('invalid timestamp or feature vector')
        if self.last_time==time and self.signature==signature:
            return self.packed()
        if self.last_time is None or reset or signature!=self.signature or time<=self.last_time or time-self.last_time>self.max_gap:
            self.frames=[];self.seen=0
        self.frames=(self.frames+[feature.clone()])[-WINDOW:]
        self.seen=min(self.seen+1,1_000_000_000);self.last_time=time;self.signature=signature
        return self.packed()

    def packed(self):
        if not self.frames:raise ValueError('window has no observations')
        x=torch.stack(self.frames);count=len(self.frames)
        valid=torch.cat((x,x.new_ones(count,1)),dim=-1)
        return torch.cat((x.new_zeros(WINDOW-count,self.features+1),valid),dim=0)
