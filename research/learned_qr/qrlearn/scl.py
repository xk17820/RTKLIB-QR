"""SCL conditional candidate loss and compact live first-order filter tape.

Forward is the real native RTKLIB. Only its active state coordinates are kept in
Torch; no frozen residual replay. Integer candidates and local H/feature inputs
are stop-gradient. No claim of derivatives through search or fix-and-hold.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from .native import array
from .model import WINDOW


def condition_position(x,P,pairs,integer):
    """Conditional position for an ACTUAL native double-difference candidate.

    `pairs` indexes the same state basis as x/P; integers are detached even when
    a caller accidentally supplies a requires_grad tensor. Qaa must be full rank.
    """
    a,b=pairs[:,0],pairs[:,1]
    Q=P[a[:,None],a[None,:]]-P[a[:,None],b[None,:]]-P[b[:,None],a[None,:]]+P[b[:,None],b[None,:]]
    cross=P[:3,a]-P[:3,b]
    return x[:3]-cross@torch.linalg.solve(Q,x[a]-x[b]-integer.detach())


class CompactTrace:
    def __init__(self,model,mode='qr'):
        self.model=model;self.mode=mode;self.device=next(model.parameters()).device
        if next(model.parameters()).dtype!=torch.float64:raise ValueError('float64 required')
        self.ids=[];self.lookup={};self.x=self.P=None;self.xp=self.Pp=None
        self.qscale=self.const([1.,1.]);self.rscale={};self.sd={};self.nfreq=3
        self.regularizers=[];self.pending_nll=None;self.fixed_position=None;self.fixed_pairs=None
        self.segment_steps=0;self.updates=0;self.candidate_count=0
        self.max_state_discrepancy=0.;self.max_cov_discrepancy=0.;self.max_candidate_discrepancy=0.
        self.last_scales={};self.smooth_terms=[]
    def const(self,x):return torch.as_tensor(x,dtype=torch.float64,device=self.device)
    def tensor(self,p,n):return self.const(array(p,n))
    def mat(self,p,n,m):return self.tensor(p,n*m).reshape(m,n).T
    @property
    def global_ids(self):return torch.tensor(self.ids,dtype=torch.long,device=self.device)
    @property
    def position(self):return self.x[:3]
    def ensure(self,ids):
        new=[int(i) for i in ids if int(i) not in self.lookup]
        # Input can contain repeated indices; preserve order only once.
        new=list(dict.fromkeys(new))
        if not new:return
        old=len(self.ids)
        self.ids.extend(new);self.lookup={v:i for i,v in enumerate(self.ids)}
        if self.x is None:self.x=self.const(np.zeros(len(self.ids)));self.P=self.const(np.zeros((len(self.ids),len(self.ids))))
        else:
            self.x=torch.cat([self.x,self.const(np.zeros(len(new)))])
            self.P=torch.nn.functional.pad(self.P,(0,len(new),0,len(new)))
    def mapids(self,ids):return torch.tensor([self.lookup[int(i)] for i in ids],dtype=torch.long,device=self.device)
    def native_values(self,e):
        x=np.ctypeslib.as_array(e.x,shape=(e.nx,))
        P=np.ctypeslib.as_array(e.P,shape=(e.nx*e.nx,)).reshape(e.nx,e.nx,order='F')
        ids=np.asarray(self.ids)
        return self.const(x[ids].copy()),self.const(P[np.ix_(ids,ids)].copy())
    def anchor(self,e,x,P):
        xn,Pn=self.native_values(e)
        dx=float((x.detach()-xn).abs().max());dp=float((P.detach()-Pn).abs().max())
        self.max_state_discrepancy=max(self.max_state_discrepancy,dx);self.max_cov_discrepancy=max(self.max_cov_discrepancy,dp)
        if dx>.01 or dp>2e-5+2e-6*float(Pn.abs().max()):
            raise RuntimeError(f'compact/native mismatch event={e.kind}: dx={dx:g}, dP={dp:g}')
        return x+(xn-x).detach(),P+(Pn-P).detach()
    @staticmethod
    def block(P,idx,value):
        out=P.clone();out[idx[:,None],idx[None,:]]=value;return out
    def detach(self):
        for n in ('x','P','xp','Pp','qscale','fixed_position'):
            v=getattr(self,n)
            if v is not None:setattr(self,n,v.detach())
        self.rscale={k:v.detach() for k,v in self.rscale.items()};self.sd={k:v.detach() for k,v in self.sd.items()}
        self.last_scales={k:v.detach() for k,v in self.last_scales.items()}
        self.regularizers=[];self.pending_nll=None;self.smooth_terms=[]
    def handle(self,e):
        k=e.kind
        if k==1:
            if e.n:
                self.ids=[];self.lookup={};self.x=self.P=None;self.segment_steps=0;self.last_scales={}
            fresh=self.x is None
            xn=array(e.x,e.nx);p=np.ctypeslib.as_array(e.P,shape=(e.nx*e.nx,))[::e.nx+1]
            self.ensure(list(range(min(9,e.nx)))+np.flatnonzero((xn!=0)|(p>0)).tolist())
            if fresh:self.x,self.P=self.native_values(e)
            else:self.x,self.P=self.anchor(e,self.x,self.P)
            self.segment_steps+=1
            self.nfreq=e.m;self.rscale={};self.sd={};self.qscale=self.const([1.,1.]);self.pending_nll=None;self.fixed_position=None
        elif k==2:
            self.ensure([e.n]);i=self.lookup[e.n];idx=self.mapids([e.n])
            self.x=self.x.index_copy(0,idx,self.tensor(e.a,1));mask=torch.ones_like(self.x);mask[i]=0
            self.P=self.P*mask[:,None]*mask[None,:];P=self.P.clone();P[i,i]=self.tensor(e.b,1)[0];self.P=P
            self.x,self.P=self.anchor(e,self.x,self.P)
        elif k==3:
            ids=array(e.ix,e.n);self.ensure(ids);idx=self.mapids(ids);F=self.mat(e.a,e.n,e.n)
            self.x=self.x.index_copy(0,idx,F@self.x[idx]);self.P=self.block(self.P,idx,F@self.P[idx[:,None],idx[None,:]]@F.T)
            self.x,self.P=self.anchor(e,self.x,self.P)
        elif k==4:
            Q=self.qscale[0]*self.mat(e.a,3,3)+self.qscale[1]*self.mat(e.b,3,3)
            ids=self.mapids([6,7,8]);self.P=self.block(self.P,ids,self.P[ids[:,None],ids[None,:]]+Q)
            self.x,self.P=self.anchor(e,self.x,self.P)
        elif k in (5,6):
            history=self.tensor(e.a,WINDOW*(e.n+1)).reshape(WINDOW,e.n+1)
            key=('q',) if k==5 else tuple(int(i) for i in array(e.ix,2))
            active=self.mode in (('q','qr') if k==5 else ('r','qr'))
            # Phase is exactly identity in SCL: avoid building unused NN graphs.
            if k==6 and getattr(self.model,'r_policy','') in ('scl','scl_blind') and history[-1,3]<.5:active=False
            scale=(self.model.q if k==5 else self.model.r)(history) if active else self.const([1.]*e.m)
            if k==5:self.qscale=scale
            else:self.rscale[key]=scale[0]
            for i,v in enumerate(scale.detach().cpu().tolist()):e.b[i]=v
            if active:
                self.regularizers.append(scale.log().square().mean())
                # Log smoothness only across continuous valid history; reset/LLI flushes history.
                if key in self.last_scales and history[-2,-1]>.5:self.smooth_terms.append((scale.log()-self.last_scales[key].log()).square().mean())
                self.last_scales[key]=scale
        elif k==7:
            key=tuple(int(i) for i in array(e.ix,2));self.sd[key]=self.tensor(e.a,1)[0]*self.rscale.get(key,self.const(e.b[0]))
        elif k==8:
            self.ensure([e.n]);i=self.lookup[e.n];P=self.P.clone();P[i,i]=P[i,i]+self.tensor(e.a,1)[0];self.P=P
        elif k==9:
            self.ensure([e.n]);self.x=self.x.index_copy(0,self.mapids([e.n]),self.const([0.]))
        elif k==10:
            ids=self.mapids(array(e.ix,e.n+e.m));part=self.x[ids[e.n:]]
            offset=self.tensor(e.a,1)[0]-(part-part.detach()).mean()
            self.x=self.x.index_copy(0,ids[:e.n],self.x[ids[:e.n]]+offset)
        elif k==11:
            self.x,self.P=self.anchor(e,self.x,self.P);self.xp,self.Pp=self.x,self.P
        elif k==12:self.filter(e)
        elif k==13:
            self.x,self.P=self.anchor(e,self.xp,self.Pp);self.updates+=1
        elif k==14:self.x,self.P=self.anchor(e,self.x,self.P)
        elif k==15:
            ids=array(e.ix,2*e.n);pairs=self.mapids(ids).reshape(e.n,2)
            pos=condition_position(self.x,self.P,pairs,self.tensor(e.a,e.n))
            discrepancy=float((pos.detach()-self.tensor(e.b,3)).abs().max())
            self.max_candidate_discrepancy=max(discrepancy,self.max_candidate_discrepancy)
            if discrepancy>.01:raise RuntimeError(f'conditional/native position mismatch {discrepancy:g}')
            self.fixed_position=pos;self.candidate_count+=1;self.fixed_pairs=pairs
        elif k==16:pass  # explicit replay-only integer audit
        else:raise RuntimeError(f'unknown SCL event {k}')
    def filter(self,e):
        self.xp,self.Pp=self.anchor(e,self.xp,self.Pp)
        n,m=e.nx,e.n
        Hnp=np.ctypeslib.as_array(e.a,shape=(n*m,)).reshape(n,m,order='F')
        H=self.const(Hnp[np.asarray(self.ids),:].T.copy());v=self.tensor(e.b,m);Rn=self.mat(e.c,m,m)
        keys={};pairs=[]
        for flag in array(e.ix,m):
            flag=int(flag);ch=((flag>>4)&15)*self.nfreq+(flag&15)
            pair=(((flag>>16)&255,ch),((flag>>8)&255,ch));pairs.append(pair)
            for key in pair:
                if key not in self.sd:raise RuntimeError(f'missing SD {key}')
                if key not in keys:keys[key]=len(keys)
        D=self.const(np.zeros((m,len(keys))))
        for j,(a,b) in enumerate(pairs):D[j,keys[a]]=1;D[j,keys[b]]=-1
        sd=torch.stack([self.sd[k] for k in keys]);Rl=(D*sd[None,:])@D.T;R=Rn+(Rl-Rl.detach())
        xn,Pn=self.native_values(e);idx=torch.nonzero((xn!=0)&(torch.diagonal(Pn)>0)).flatten()
        xa=self.xp[idx];Pa=self.Pp[idx[:,None],idx[None,:]];Ha=H[:,idx];Pa=(Pa+Pa.T)*.5
        residual=v-H@(self.xp-xn);PH=Pa@Ha.T;S=Ha@PH+R;L=torch.linalg.cholesky((S+S.T)*.5)
        K=torch.cholesky_solve(PH.T,L).T;self.xp=self.xp.index_copy(0,idx,xa+K@residual)
        A=torch.eye(len(idx),dtype=Pa.dtype,device=Pa.device)-K@Ha
        post=A@Pa@A.T+K@R@K.T;self.Pp=self.block(self.Pp,idx,(post+post.T)*.5)
        white=torch.linalg.solve_triangular(L,residual[:,None],upper=False)
        self.pending_nll=(white.square().sum()+2*L.diagonal().log().sum()+m*math.log(2*math.pi))*.5/m
