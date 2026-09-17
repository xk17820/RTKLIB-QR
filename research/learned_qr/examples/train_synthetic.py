#!/usr/bin/env python3
"""Synthetic NON-GNSS smoke test of alternating Q/R learning through a filter.

This is not RINEX training and produces NO validated RTK weights. Its exported
provenance is always `synthetic`, which native deployment refuses by default.
Run from the kit root: python -m examples.train_synthetic --epochs 20
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from qrlearn.model import QRModel
from qrlearn.features import TemporalWindow,q_features,r_features
from qrlearn.filtering import predict,dd_covariance,ekf_update

def make_data(length: int,seed: int):
    rng=torch.Generator().manual_seed(seed);d=torch.float64
    t=torch.arange(length,dtype=d)
    reference=.1*t+torch.sin(.35*t)
    sd_truth=reference[:,None]*torch.tensor([1.,0.,-1.],dtype=d)
    sigma=torch.full((length,3),.2,dtype=d)
    sigma[(torch.arange(length)%9)<3,0]=2.0
    measurements=sd_truth+torch.randn(length,3,generator=rng,dtype=d)*sigma
    cn0=44-12*(sigma>.3).to(d)
    return reference,measurements,cn0

def sequence_loss(model,data):
    truth,sd_observations,cn0=data;d=torch.float64
    D=torch.tensor([[1.,-1.,0.],[1.,0.,-1.]],dtype=d)
    H=D@torch.tensor([[1.,0.],[0.,0.],[-1.,0.]],dtype=d)
    F=torch.tensor([[1.,1.],[0.,1.]],dtype=d)
    x=torch.zeros(2,dtype=d);P=torch.eye(2,dtype=d)*2
    qw=TemporalWindow(10);rw=[TemporalWindow(12) for _ in range(3)]
    loss=x.new_tensor(0.)
    for k in range(truth.numel()):
        xp=F@x;P0=F@P@F.mT
        vel=torch.stack((xp[1],xp[1]*0,xp[1]*0))
        qf=q_features(1.,vel,vel*0,P0[0,0],3,2)
        qscale=model.q(qw.push(float(k),0,qf))
        histories=[]
        for j in range(3):
            previous=(float(cn0[k-1,j])-40)/10 if k else None
            rf=r_features(float(cn0[k,j]),44,.8,True,0,3,0,100,0,0,False,previous,k+1)
            histories.append(rw[j].push(float(k),j,rf))
        rscale=model.r(torch.stack(histories)).squeeze(-1)
        # Toy two-state driving noise, NOT RTKLIB's PVA/ambiguity dynamics.
        Q=torch.diag(x.new_tensor([.01,.03])*qscale)
        xp,Pp=predict(x,P,F,Q)
        R=dd_covariance(D,x.new_full((3,),.04)*rscale)
        x,P,nis,nll=ekf_update(xp,Pp,D@sd_observations[k],lambda state:H@state,H,R)
        loss=loss+(x[0]-truth[k]).square()+.002*nll+.0001*(qscale.log().square().mean()+rscale.log().square().mean())
    return loss/truth.numel()

def train(epochs: int=20,output: str|Path='outputs/synthetic.qr',length: int=40):
    if epochs<2 or length<5:raise ValueError('at least two alternating stages and five epochs per sequence required')
    torch.manual_seed(20260917);torch.set_num_threads(1)
    model=QRModel().double();data=[make_data(length,s) for s in (11,23)]
    objective=lambda:sum(sequence_loss(model,item) for item in data)/len(data)
    with torch.no_grad():initial=float(objective())
    q0=model.q.fc2.weight.detach().clone();r0=model.r.fc2.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    history=[]
    for step in range(epochs):
        stage='r' if step%2==0 else 'q'
        for p in model.q.parameters():p.requires_grad_(stage=='q')
        for p in model.r.parameters():p.requires_grad_(stage=='r')
        optimizer.zero_grad(set_to_none=True);loss=objective()
        if not torch.isfinite(loss):raise RuntimeError('non-finite synthetic training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step();history.append({'stage':stage,'loss_before_update':float(loss.detach())})
    with torch.no_grad():final=float(objective())
    output=Path(output);model.export(output,provenance='synthetic')
    report={'data_kind':'synthetic_non_GNSS','purpose':'gradient/export smoke test ONLY',
            'initial_training_loss':initial,'final_training_loss':final,
            'q_head_changed':bool((model.q.fc2.weight.detach()-q0).abs().sum()>0),
            'r_head_changed':bool((model.r.fc2.weight.detach()-r0).abs().sum()>0),
            'epochs':epochs,'sequence_length':length,'stages':history,
            'real_GNSS_training_completed':False,'RTK_positioning_gain_established':False}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--length',type=int,default=40)
    p.add_argument('--output',type=Path,default=Path('outputs/synthetic.qr'))
    a=p.parse_args();print(json.dumps(train(a.epochs,a.output,a.length),indent=2))
if __name__=='__main__':main()
