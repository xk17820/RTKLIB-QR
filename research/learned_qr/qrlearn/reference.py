"""Strict GPST/ECEF reference data, ENU supervision and transparent metrics.

GPST seconds mean seconds since 1980-01-06, NOT Unix seconds. Input coordinates
must already refer to the GNSS antenna point and the solver's terrestrial frame.
This module never passes reference positions into native GNSS features or states.
"""
from __future__ import annotations
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import numpy as np
import torch

@dataclass
class Reference:
    time: np.ndarray
    xyz: np.ndarray
    valid: np.ndarray
    max_gap: float=1.0

    @classmethod
    def load(cls,path,*,max_gap=1.0,time_offset=0.0):
        if not math.isfinite(max_gap) or max_gap<=0 or not math.isfinite(time_offset):
            raise ValueError('invalid reference max_gap/time_offset')
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);names=set(reader.fieldnames or ())
            if not {'x_m','y_m','z_m','valid'}<=names:
                raise ValueError('reference needs x_m,y_m,z_m,valid columns (ECEF metres)')
            week_tow={'gps_week','tow_s'}<=names
            if not week_tow and 't_gpst_s' not in names:
                raise ValueError('reference needs gps_week,tow_s or t_gpst_s (GPS epoch, not Unix)')
            rows=[]
            for line,row in enumerate(reader,start=2):
                try:
                    valid=float(row['valid'])
                    if valid not in (0,1):raise ValueError('valid must be 0 or 1')
                    if week_tow:
                        week=float(row['gps_week']);tow=float(row['tow_s'])
                        if not week.is_integer() or week<0 or not 0<=tow<604800:
                            raise ValueError('invalid GPS week or TOW')
                        time=week*604800+tow
                    else:time=float(row['t_gpst_s'])
                    if not math.isfinite(time) or time<0:raise ValueError('invalid GPST')
                    xyz=[float(row[k]) if row[k].strip() else float('nan') for k in ('x_m','y_m','z_m')]
                    if valid and (not np.isfinite(xyz).all() or not 3e6<np.linalg.norm(xyz)<1e7):
                        raise ValueError('valid reference must be plausible ECEF coordinates in metres')
                    rows.append([time+time_offset,*xyz,valid])
                except (TypeError,ValueError,KeyError) as exc:
                    raise ValueError(f'{path}:{line}: {exc}') from exc
        if not rows:raise ValueError('empty reference file')
        data=np.asarray(rows,dtype=np.float64)
        if np.any(np.diff(data[:,0])<=0):raise ValueError('reference times must be strictly increasing without duplicates')
        return cls(data[:,0],data[:,1:4],data[:,4].astype(bool),max_gap)

    def at(self,time):
        if not math.isfinite(time):return np.full(3,np.nan),False
        i=int(np.searchsorted(self.time,time))
        for j in (i,i-1):
            if 0<=j<len(self.time) and abs(self.time[j]-time)<=1e-6:
                return self.xyz[j].copy(),bool(self.valid[j])
        if i==0 or i==len(self.time):return np.full(3,np.nan),False
        if not self.valid[i-1] or not self.valid[i] or self.time[i]-self.time[i-1]>self.max_gap:
            return np.full(3,np.nan),False
        alpha=(time-self.time[i-1])/(self.time[i]-self.time[i-1])
        return self.xyz[i-1]*(1-alpha)+self.xyz[i]*alpha,True

def enu_rotation(xyz)->np.ndarray:
    x,y,z=np.asarray(xyz,dtype=np.float64)
    if not np.isfinite([x,y,z]).all():raise ValueError('nonfinite reference')
    e2=6.6943799901413165e-3;a=6378137.0;r=math.hypot(x,y)
    lon=math.atan2(y,x);lat=math.atan2(z,r*(1-e2))
    for _ in range(8):
        N=a/math.sqrt(1-e2*math.sin(lat)**2)
        lat=math.atan2(z+e2*N*math.sin(lat),r)
    sl,cl=math.sin(lon),math.cos(lon);sp,cp=math.sin(lat),math.cos(lat)
    return np.array([[-sl,cl,0],[-sp*cl,-sp*sl,cp],[cp*cl,cp*sl,sp]])

def enu_error(position:torch.Tensor,reference)->torch.Tensor:
    if position.shape!=(3,):raise ValueError('position must have three ECEF components')
    ref=torch.as_tensor(reference,dtype=position.dtype,device=position.device)
    rotation=torch.as_tensor(enu_rotation(reference),dtype=position.dtype,device=position.device)
    return rotation@(position-ref)

def position_loss(position,reference,*,up_weight=1.0,huber_delta=1.0):
    if not math.isfinite(up_weight) or up_weight<=0 or not math.isfinite(huber_delta) or huber_delta<0:
        raise ValueError('invalid loss weights')
    e=enu_error(position,reference)
    if huber_delta:
        a=e.abs();cost=torch.where(a<=huber_delta,.5*e.square(),huber_delta*(a-.5*huber_delta))
    else:cost=e.square()
    return (cost*cost.new_tensor([1.,1.,up_weight])).sum()

def summarize(errors,status,*,attempted,fix_threshold=.1):
    e=np.asarray(errors,dtype=float).reshape(-1,3);status=np.asarray(status,dtype=int)
    if len(e)!=len(status) or not np.isfinite(e).all() or attempted<len(e) or fix_threshold<=0:
        raise ValueError('invalid evaluation arrays/threshold')
    n=len(e);fixed=status==1
    out={'attempted_epochs':int(attempted),'matched_epochs':n,
         'matched_fraction':n/attempted if attempted else 0.,
         'fixed_matched_epochs':int(fixed.sum()),
         'fixed_fraction_of_matched':float(fixed.mean()) if n else 0.,
         'fixed_error_threshold_m':fix_threshold}
    for label,mask in [('',np.ones(n,dtype=bool)),('float_',status==2),('fix_',fixed)]:
        a=e[mask]
        if not len(a):
            out.update({label+k:None for k in ('rms_h_m','rms_u_m','rms_3d_m','p95_3d_m','p99_3d_m','max_3d_m')});continue
        distance=np.linalg.norm(a,axis=1)
        out.update({label+'rms_h_m':float(np.sqrt(np.mean(np.sum(a[:,:2]**2,axis=1)))),
                    label+'rms_u_m':float(np.sqrt(np.mean(a[:,2]**2))),
                    label+'rms_3d_m':float(np.sqrt(np.mean(distance**2))),
                    label+'p95_3d_m':float(np.quantile(distance,.95)),
                    label+'p99_3d_m':float(np.quantile(distance,.99)),
                    label+'max_3d_m':float(distance.max())})
    bad=int((np.linalg.norm(e[fixed],axis=1)>fix_threshold).sum())
    out['fixed_large_error_proxy_count']=bad
    out['fixed_large_error_proxy_fraction']=bad/int(fixed.sum()) if fixed.any() else None
    out['proxy_note']='Reference position threshold exceedance is NOT proof of incorrect integer ambiguities.'
    return out
