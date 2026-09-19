"""Differentiable tape for the LIVE native FLOAT recursion.

The forward pass is native RTKLIB, including fresh GNSS residuals at every step.
Gradients are CONDITIONAL FIRST-ORDER EKF gradients: selection/gating, local H,
feature extraction and coordinate rotations are detached. There is no gradient
through LAMBDA. Native state resets sever the corresponding state dependencies.
This is not exact autodiff through every C instruction; see docs/TRAINING.md.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from .model import QRModel,WINDOW
from .native import array

class TraceEngine:
    def __init__(self,model:QRModel,mode='qr'):
        if mode not in ('off','q','r','qr'):raise ValueError('invalid Q/R mode')
        p=next(model.parameters())
        if p.dtype!=torch.float64:raise ValueError('RTK training requires float64 model')
        self.model=model;self.mode=mode;self.device=p.device
        self.x=self.P=self.xp=self.Pp=None
        self.qscale=torch.ones(2,dtype=torch.float64,device=self.device)
        self.rscale={};self.sd={};self.nfreq=3
        self.nll=[];self.pending_nll=None;self.regularizers=[]
        self.segment_steps=0;self.updates=0;self.max_state_discrepancy=0.0;self.max_cov_discrepancy=0.0

    def tensor(self,p,n):
        return torch.as_tensor(array(p,n),dtype=torch.float64,device=self.device)
    def matrix(self,p,n,m):
        return self.tensor(p,n*m).reshape(m,n).mT
    def constant(self,x):return torch.as_tensor(x,dtype=torch.float64,device=self.device)
    def indices(self,p,n):return torch.as_tensor(array(p,n),dtype=torch.long,device=self.device)

    def anchor(self,x,P,e):
        native_x=self.tensor(e.x,e.nx);native_P=self.matrix(e.P,e.nx,e.nx)
        if x is None:return native_x,native_P
        dx=float((x.detach()-native_x).abs().max());dp=float((P.detach()-native_P).abs().max())
        self.max_state_discrepancy=max(self.max_state_discrepancy,dx)
        self.max_cov_discrepancy=max(self.max_cov_discrepancy,dp)
        # Allow floating point differences from native versus Torch floating-point solves, not
        # missing state operations. Native values stay authoritative for next step.
        tolP=2e-5+2e-6*float(native_P.abs().max())
        if dx>0.01 or dp>tolP:
            raise RuntimeError(f'native/tape mismatch at event {e.kind}: x={dx:g}, P={dp:g}, tolerance={tolP:g}')
        return x+(native_x-x).detach(),P+(native_P-P).detach()

    @staticmethod
    def block(P,idx,value):
        out=P.clone();out[idx[:,None],idx[None,:]]=value;return out

    @property
    def position(self):
        if self.x is None:raise RuntimeError('no FLOAT state')
        return self.x[:3]

    def detach(self):
        for name in ('x','P','xp','Pp','qscale'):
            value=getattr(self,name)
            if value is not None:setattr(self,name,value.detach())
        self.rscale={k:v.detach() for k,v in self.rscale.items()}
        self.sd={k:v.detach() for k,v in self.sd.items()}
        self.nll=[];self.pending_nll=None;self.regularizers=[]

    def handle(self,e):
        kind=e.kind
        if kind==1: # BEGIN (n=1 means full native reset)
            if e.n:
                self.x=self.P=None;self.segment_steps=0
            self.segment_steps+=1
            self.x,self.P=self.anchor(self.x,self.P,e)
            self.nfreq=e.m;self.rscale={};self.sd={}
            self.qscale=self.constant([1.,1.]);self.pending_nll=None
        elif kind==2: # scalar initialization/reset; covariance row+column erased
            i=e.n;xi=self.tensor(e.a,1);variance=self.tensor(e.b,1)[0]
            idx=torch.tensor([i],device=self.device)
            self.x=self.x.index_copy(0,idx,xi)
            mask=torch.ones_like(self.x);mask[i]=0
            self.P=self.P*mask[:,None]*mask[None,:]
            out=self.P.clone();out[i,i]=variance;self.P=out
            self.x,self.P=self.anchor(self.x,self.P,e)
        elif kind==3: # native compact position transition
            idx=self.indices(e.ix,e.n);F=self.matrix(e.a,e.n,e.n)
            self.x=self.x.index_copy(0,idx,F@self.x[idx])
            self.P=self.block(self.P,idx,F@self.P[idx[:,None],idx[None,:]]@F.mT)
            self.x,self.P=self.anchor(self.x,self.P,e)
        elif kind==4: # acceleration Q; local rotation frozen at this native state
            Q=self.qscale[0]*self.matrix(e.a,3,3)+self.qscale[1]*self.matrix(e.b,3,3)
            idx=torch.arange(6,9,device=self.device)
            self.P=self.block(self.P,idx,self.P[6:9,6:9]+Q)
            self.x,self.P=self.anchor(self.x,self.P,e)
        elif kind in (5,6): # causal packed features; explicit stop-gradient
            history=self.tensor(e.a,WINDOW*(e.n+1)).reshape(WINDOW,e.n+1)
            active=(kind==5 and self.mode in ('q','qr')) or (kind==6 and self.mode in ('r','qr'))
            net=self.model.q if kind==5 else self.model.r
            scale=net(history) if active else self.constant([1.]*e.m)
            if kind==5:self.qscale=scale
            else:self.rscale[tuple(int(v) for v in array(e.ix,2))]=scale[0]
            for i,value in enumerate(scale.detach().cpu().tolist()):e.b[i]=value
            if active:self.regularizers.append(torch.log(scale).square().mean())
        elif kind==7:
            key=tuple(int(v) for v in array(e.ix,2))
            self.sd[key]=self.tensor(e.a,1)[0]*self.rscale.get(key,self.constant(float(e.b[0])))
        elif kind==8:
            out=self.P.clone();out[e.n,e.n]=out[e.n,e.n]+self.tensor(e.a,1)[0];self.P=out
        elif kind==9:
            self.x=self.x.index_copy(0,torch.tensor([e.n],device=self.device),self.constant([0.]))
        elif kind==10: # mean(bias - old ambiguity), x only, exactly as native
            ids=self.indices(e.ix,e.n+e.m);contributors=self.x[ids[e.n:]]
            if e.m<1:raise RuntimeError('ambiguity offset without contributors')
            offset=self.tensor(e.a,1)[0]-(contributors-contributors.detach()).mean()
            self.x=self.x.index_copy(0,ids[:e.n],self.x[ids[:e.n]]+offset)
        elif kind==11:
            self.x,self.P=self.anchor(self.x,self.P,e)
            self.xp,self.Pp=self.x,self.P
        elif kind==12:self.filter(e)
        elif kind==13:
            self.x,self.P=self.anchor(self.xp,self.Pp,e)
            if self.pending_nll is not None:self.nll.append(self.pending_nll)
            self.updates+=1
        elif kind==14:
            self.x,self.P=self.anchor(self.x,self.P,e)
        else:raise RuntimeError(f'unknown native callback event {kind}')

    def filter(self,e):
        self.xp,self.Pp=self.anchor(self.xp,self.Pp,e)
        n,m=e.nx,e.n
        H=self.matrix(e.a,n,m).mT;v=self.tensor(e.b,m);Rn=self.matrix(e.c,m,m)
        flags=[int(f) for f in array(e.ix,m)]
        pairs=[];keys={}
        for flag in flags:
            channel=((flag>>4)&15)*self.nfreq+(flag&15)
            if ((flag>>4)&15)>1:raise RuntimeError('unsupported non-GNSS constraint in training')
            pair=(((flag>>16)&255,channel),((flag>>8)&255,channel));pairs.append(pair)
            for key in pair:
                if key not in self.sd:raise RuntimeError(f'missing SD variance for {key}')
                if key not in keys:keys[key]=len(keys)
        D=torch.zeros((m,len(keys)),dtype=torch.float64,device=self.device)
        for j,(a,b) in enumerate(pairs):D[j,keys[a]]=1;D[j,keys[b]]=-1
        sd=torch.stack([self.sd[k] for k in keys]);Rlearn=(D*sd[None,:])@D.mT
        # Preserve native half-cycle additions and exact forward rounding. The
        # derivative includes correlated reference-satellite SD contributions.
        R=Rn+(Rlearn-Rlearn.detach())
        xn=self.tensor(e.x,n);Pn=self.matrix(e.P,n,n)
        idx=torch.nonzero((xn!=0)&(torch.diagonal(Pn)>0)).flatten()
        xa=self.xp[idx];Pa=self.Pp[idx[:,None],idx[None,:]];Ha=H[:,idx]
        residual=v-H@(self.xp-xn) # local dv/dx=-H; native relinearizes every epoch
        Pa=(Pa+Pa.mT)*.5
        PH=Pa@Ha.mT;S=Ha@PH+R
        L=torch.linalg.cholesky((S+S.mT)*.5)
        K=torch.cholesky_solve(PH.mT,L).mT
        self.xp=self.xp.index_copy(0,idx,xa+K@residual)
        A=torch.eye(len(idx),dtype=Pa.dtype,device=Pa.device)-K@Ha
        posterior=A@Pa@A.mT+K@R@K.mT
        posterior=(posterior+posterior.mT)*.5
        self.Pp=self.block(self.Pp,idx,posterior)
        L=torch.linalg.cholesky((S+S.mT)*.5)
        white=torch.linalg.solve_triangular(L,residual[:,None],upper=False)
        self.pending_nll=(white.square().sum()+2*torch.log(torch.diagonal(L)).sum()+m*math.log(2*math.pi))*.5/m
