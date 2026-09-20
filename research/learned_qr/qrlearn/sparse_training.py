"""Deterministic rotating-window TBPTT with ALL raw epochs propagated in C.

Inactive windows are NOT omitted from the trajectory. Current weights are copied
into C without resetting state/history. Only gradients are sampled and truncated.
This is stochastic truncated BPTT, not full-sequence exact differentiation.
"""
from pathlib import Path
import math,tempfile,time
import numpy as np
import torch
from .native import Session
from .scl import CompactTrace
from .reference import Reference,enu_rotation,position_loss,summarize
from .training import has_filter_state


def sparse_pass(model,routes,args,*,optimizer=None):
    train=optimizer is not None;start=time.perf_counter()
    es=[];ss=[];n=matches=steps=diffslots=ncandidate=0;maxdx=maxdp=0.;grad=[];objectives=[]
    with tempfile.TemporaryDirectory(prefix='scl-live-') as tmp:
      file=Path(tmp)/'weights.qr';model.export(file,provenance='synthetic',max_gap=args.max_gap)
      for route_index,r in enumerate(routes):
        ref=Reference.load(r['reference'],max_gap=r.get('reference_max_gap_s',args.reference_max_gap),time_offset=r.get('reference_time_offset_s',0))
        with Session(config=r.get('config',args.config),rover=r['rover'],base=r['base'],nav=r['nav'],model=file,
          mode=args.mode,training=True,allow_test=True,max_gap=args.max_gap,pair_age=args.pair_age,pairing_latency=getattr(args,"pairing_latency",None),library=args.library) as session:
          k=0;engine=None;losses=[];nll=[];last_time=None;since_start=0
          def flush():
            nonlocal steps,maxdx,maxdp
            if engine is None:return
            if losses:
                loss=torch.stack(losses).mean()
                if nll and args.nll_weight:loss=loss+args.nll_weight*torch.stack(nll).mean()
                if engine.regularizers and args.reg_weight:loss=loss+args.reg_weight*torch.stack(engine.regularizers).mean()
                if engine.smooth_terms and args.smooth_weight:loss=loss+args.smooth_weight*torch.stack(engine.smooth_terms).mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite SCL objective')
                model.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.gradient_clip,error_if_nonfinite=True)
                grad.append(float(norm));objectives.append(float(loss.detach()));engine.detach();optimizer.step();steps+=1
                model.export(file,provenance='synthetic',max_gap=args.max_gap)
            maxdx=max(maxdx,engine.max_state_discrepancy);maxdp=max(maxdp,engine.max_cov_discrepancy)
            session.reload_model(file,allow_test=True,candidates=False);losses.clear();nll.clear()
          while not args.max_epochs_per_route or k<args.max_epochs_per_route:
            if k%args.sequence_length==0:
                block=k//args.sequence_length
                active=train and (block+getattr(args,'round_index',0)+route_index)%args.gradient_stride==0
                engine=CompactTrace(model,args.mode) if active else None
                if active:session.reload_model(None,candidates=args.conditional_weight>0)
            rec=session.step(engine.handle if engine else None)
            if rec is None:break
            n+=1;k+=1
            if last_time is None or rec['time']-last_time>args.max_gap:since_start=0
            since_start+=1;last_time=rec['time']
            target,valid=ref.at(rec['time'])
            if valid and has_filter_state(rec['status']) and since_start>args.warmup_epochs:
                pos=engine.position if engine else None
                xy=pos.detach().cpu().numpy() if pos is not None else rec['position']
                es.append(enu_rotation(target)@(xy-target));ss.append(rec['status']);matches+=1
                if engine:
                    val=position_loss(pos,target,up_weight=args.up_weight,huber_delta=args.huber_delta)
                    if args.conditional_weight>0 and engine.fixed_position is not None:
                        val=val+args.conditional_weight*position_loss(engine.fixed_position,target,up_weight=args.up_weight,huber_delta=args.huber_delta);ncandidate+=1
                    losses.append(val);diffslots+=1
                    if engine.pending_nll is not None:nll.append(engine.pending_nll)
            if k%args.sequence_length==0:flush();engine=None
          flush()
        print(f'  pass {r["id"]} raw={k} cumulative_gradient_epochs={diffslots}',flush=True)
    if matches==0:raise RuntimeError('no matched continuous-filter reference')
    a=summarize(es,ss,attempted=n)
    e=np.asarray(es);a.update(selection_weighted_rmse_m=float(np.sqrt(np.mean(e[:,0]**2+e[:,1]**2+args.up_weight*e[:,2]**2))),
      optimizer_steps=steps,conditional_supervised_epochs=ncandidate,gradient_epochs=diffslots,raw_forward_epochs=n,
      max_native_tape_state_difference=maxdx,max_native_tape_cov_difference=maxdp,
      mean_preclip_gradient_norm=float(np.mean(grad)) if grad else None,
      mean_training_objective=float(np.mean(objectives)) if objectives else None,wall_seconds=time.perf_counter()-start)
    return a
