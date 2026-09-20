"""Reference-supervised local training with raw native RINEX passes and TBPTT."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .model import QRModel
from .native import Session
from .live import TraceEngine
from .scl import CompactTrace
from .selection import native_validation,deployment_decision
from .reference import Reference,enu_rotation,position_loss,summarize

def load_manifest(path,*,allow_training_only=False):
    path=Path(path).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not data.get('train'):raise ValueError('manifest needs a nonempty train list')
    if not data.get('validation') and not allow_training_only:
        raise ValueError('manifest needs held-out validation routes; explicitly use --allow-training-only for plumbing/debug')
    result={};seen_ids=set()
    for split in ('train','validation'):
        result[split]=[]
        for source in data.get(split,[]):
            if not isinstance(source,dict) or not {'id','rover','base','nav','reference'}<=source.keys():
                raise ValueError('each route needs id,rover,base,nav,reference')
            route=dict(source)
            if not isinstance(route['id'],str) or not route['id'] or route['id'] in seen_ids:
                raise ValueError('route IDs must be nonempty and unique')
            seen_ids.add(route['id'])
            if not isinstance(route['nav'],list) or not route['nav']:raise ValueError('nav must be a nonempty list')
            def absolute(p):return (path.parent/Path(p).expanduser()).resolve()
            for name in ('rover','base','reference','config'):
                if name in route:route[name]=absolute(route[name])
            route['nav']=[absolute(p) for p in route['nav']]
            result[split].append(route)
    train={p['rover'] for p in result['train']};validation={p['rover'] for p in result['validation']}
    if train&validation:raise ValueError('training/validation rover files overlap; use independent routes')
    return result

def checkpoint_save(path,model,*,max_gap,epoch,mode):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save({'format':2,'r_policy':model.r_policy,'guard_anchor':getattr(model.r,'anchor',40.),
        'guard_width':getattr(model.r,'width',10.),'phase_guard_cap':getattr(model.r,'phase_cap',1.),'evidence_threshold':getattr(model.r,'anchor',2.),'evidence_width':getattr(model.r,'width',4.),'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'q_limit':model.q.limit,'r_limit':model.r.limit,'max_gap':max_gap,'epoch':epoch,'mode':mode},tmp)
    tmp.replace(path)

def checkpoint_load(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(data,dict) or data.get('format') not in (1,2):raise ValueError('unsupported checkpoint format')
    model=QRModel(data['q_limit'],data['r_limit'],r_policy=data.get('r_policy','unconstrained'),
        guard_anchor=data.get('guard_anchor',40.),guard_width=data.get('guard_width',10.),
        phase_guard_cap=data.get('phase_guard_cap',1.),evidence_threshold=data.get('evidence_threshold',2.),evidence_width=data.get('evidence_width',4.)).double().to(device)
    model.load_state_dict(data['state_dict'],strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('checkpoint contains nonfinite parameters')
    return model,data

def dataset_pass(model,routes,args,*,optimizer=None):
    if getattr(args,"gradient_stride",1)>1:
        from .sparse_training import sparse_pass
        return sparse_pass(model,routes,args,optimizer=optimizer)
    training=optimizer is not None
    errors=[];statuses=[];attempted=matched=steps=updates=0;candidate_epochs=0;max_dx=max_dp=0.;grad_norms=[];loss_values=[]
    for route in routes:
        reference=Reference.load(route['reference'],max_gap=float(route.get('reference_max_gap_s',args.reference_max_gap)),
                                 time_offset=float(route.get('reference_time_offset_s',0)))
        engine=CompactTrace(model,args.mode) if getattr(args,'compact',False) or model.r_policy in ('scl','scl_blind','scl_unconstrained') else TraceEngine(model,args.mode);losses=[];nlls=[]
        def flush():
            nonlocal steps
            if training and losses:
                loss=torch.stack(losses).mean()
                if nlls and args.nll_weight:loss=loss+args.nll_weight*torch.stack(nlls).mean()
                if engine.regularizers and args.reg_weight:
                    loss=loss+args.reg_weight*torch.stack(engine.regularizers).mean()
                if getattr(args,'smooth_weight',0) and getattr(engine,'smooth_terms',[]):
                    loss=loss+args.smooth_weight*torch.stack(engine.smooth_terms).mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                model.zero_grad(set_to_none=True);loss.backward()
                parameters=[p for p in model.parameters() if p.requires_grad]
                norm=torch.nn.utils.clip_grad_norm_(parameters,args.gradient_clip,error_if_nonfinite=True)
                grad_norms.append(float(norm));loss_values.append(float(loss.detach()))
                engine.detach();optimizer.step();steps+=1
            else:engine.detach()
            losses.clear();nlls.clear()
        with Session(config=route.get('config',args.config),rover=route['rover'],base=route['base'],nav=route['nav'],
                     mode=args.mode,training=True,max_gap=args.max_gap,pair_age=args.pair_age,pairing_latency=getattr(args,"pairing_latency",None),library=args.library,scl=model.r_policy in ('scl','scl_blind','scl_unconstrained'),
                     candidates=getattr(args,'conditional_weight',0)>0) as session:
            with torch.set_grad_enabled(training):
                number=0
                while not args.max_epochs_per_route or number<args.max_epochs_per_route:
                    record=session.step(engine.handle)
                    if record is None:break
                    attempted+=1;number+=1
                    target,valid=reference.at(record['time'])
                    if valid and has_filter_state(record['status']) and engine.segment_steps>args.warmup_epochs:
                        error=enu_rotation(target)@(engine.position.detach().cpu().numpy()-target)
                        errors.append(error);statuses.append(record['status']);matched+=1
                        if training:
                            losses.append(position_loss(engine.position,target,up_weight=args.up_weight,huber_delta=args.huber_delta))
                            if getattr(args,'conditional_weight',0)>0 and engine.fixed_position is not None:
                                losses[-1]=losses[-1]+args.conditional_weight*position_loss(engine.fixed_position,target,up_weight=args.up_weight,huber_delta=args.huber_delta)
                                candidate_epochs+=1
                            if engine.pending_nll is not None:nlls.append(engine.pending_nll)
                    if number%args.sequence_length==0:flush()
                flush()
        updates+=engine.updates;max_dx=max(max_dx,engine.max_state_discrepancy);max_dp=max(max_dp,engine.max_cov_discrepancy)
    if not matched:
        raise ValueError('zero supervised FLOAT epochs: check GPST/frame, reference validity/overlap, warmup and raw observations')
    metrics=summarize(errors,statuses,attempted=attempted)
    metrics.update(conditional_supervised_epochs=candidate_epochs,optimizer_steps=steps,native_float_updates=updates,
                   max_native_tape_state_difference=max_dx,max_native_tape_cov_difference=max_dp,
                   mean_preclip_gradient_norm=float(np.mean(grad_norms)) if grad_norms else None,
                   mean_training_objective=float(np.mean(loss_values)) if loss_values else None)
    e=np.asarray(errors)
    metrics['selection_weighted_rmse_m']=float(np.sqrt(np.mean(e[:,0]**2+e[:,1]**2+args.up_weight*e[:,2]**2)))
    return metrics

def has_filter_state(status):
    """States exported by the continuous native relative filter.

    RTKLIB labels a successful update with fewer than four valid phase satellites
    as SOLQ_DGPS (4). Its position is still rtk->x and has a differentiable tape.
    SOLQ_SINGLE (5) is independent SPP and must NOT be given a fictitious gradient.
    """
    return int(status) in (2,4)


def train(args):
    for name in ('epochs','sequence_length','threads'):
        if getattr(args,name)<1:raise ValueError(f'{name} must be positive')
    if args.warmup_epochs<0 or args.max_epochs_per_route<0:raise ValueError('epoch limits cannot be negative')
    for name in ('learning_rate','gradient_clip','up_weight','max_gap','reference_max_gap'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:raise ValueError(f'{name} must be positive and finite')
    if getattr(args,'gradient_stride',1)<1:raise ValueError('gradient_stride must be positive')
    for name in ('nll_weight','reg_weight','huber_delta','conditional_weight','smooth_weight'):
        if not math.isfinite(getattr(args,name,0)) or getattr(args,name,0)<0:raise ValueError(f'{name} cannot be negative/nonfinite')
    if not math.isfinite(args.selection_min_improvement) or not 0<=args.selection_min_improvement<1:
        raise ValueError('selection minimum must be in [0,1)')
    if not math.isfinite(args.selection_fix_threshold) or args.selection_fix_threshold<=0:
        raise ValueError('selection fix threshold must be positive')
    routes=load_manifest(args.manifest,allow_training_only=args.allow_training_only)
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    if args.init:
        model,meta=checkpoint_load(args.init,args.device)
        if meta['max_gap']!=args.max_gap:raise ValueError('--max-gap differs from initialized checkpoint')
        if args.phase_guard_cap is not None:
            if model.r_policy!='code_guard' or not math.isfinite(args.phase_guard_cap) or not 1<=args.phase_guard_cap<=100:
                raise ValueError('phase guard override needs code_guard and a cap in [1,100]')
            model.r.phase_cap=args.phase_guard_cap
    else:model=QRModel(args.q_limit,args.r_limit,r_policy=args.r_policy,
        guard_anchor=args.guard_anchor,guard_width=args.guard_width,
        phase_guard_cap=args.phase_guard_cap if args.phase_guard_cap is not None else 1.,
        evidence_threshold=getattr(args,'evidence_threshold',2.),evidence_width=getattr(args,'evidence_width',4.)).double().to(args.device)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    optimizers={name:torch.optim.Adam(getattr(model,name).parameters(),lr=args.learning_rate) for name in ('q','r')}
    metadata={'method':'live native conditional-first-order FLOAT differentiation',
        'gradient_stops':['discrete selection/gating/reset choices','local H','causal features','coordinate rotations'],
        'numerics':'native Cholesky/Joseph; compare against stable off baseline as well as legacy',
        'reference_role':'supervision only; never passed to native inference',
        'training_ambiguity_resolution':'Continuous FLOAT/DGPS native filter states; SINGLE fallback excluded;  read-only native LAMBDA candidate auxiliary when enabled; original AR/hold retained in replay',
        'auxiliary_gradient':'fixed-candidate conditional derivative; NOT derivative of integer search/hold switching',
        'selection':('native complete-RTK gate against identity baseline' if args.selection=='rtk-safe' else 'held-out FLOAT validation') if routes['validation'] else 'training-only: no generalization evidence',
        'provenance':args.provenance,'arguments':vars(args),
        'manifest':json.loads(Path(args.manifest).read_text(encoding='utf-8'))}
    # argparse handlers are not serialized; no callable/pickle payload in metadata.
    metadata['arguments']={k:v for k,v in vars(args).items() if not callable(v)}
    (output/'run.json').write_text(json.dumps(metadata,indent=2,default=str,allow_nan=False)+'\n')
    safe=args.selection=='rtk-safe' and bool(routes['validation'])
    best=float('inf');baseline=None;selected_epoch=None
    if safe:
        # Identity competition is the ORIGINAL R/Q, including phase cap=1.
        identity=QRModel(model.q.limit,model.r.limit,r_policy=model.r_policy,
            guard_anchor=getattr(model.r,'anchor',40.),guard_width=getattr(model.r,'width',10.)).double().to(args.device)
        baseline=native_validation(identity,routes['validation'],args,output/'baseline.qr',baseline=True)
        best=baseline['aggregate']['rms_3d_m'];selected_epoch=0
        if best is None or not math.isfinite(best):raise ValueError('no valid native validation baseline')
        (output/'validation-baseline.json').write_text(json.dumps(baseline,indent=2,allow_nan=False)+'\n')
        checkpoint_save(output/'best.pt',identity,max_gap=args.max_gap,epoch=0,mode=args.mode)
        identity.export(output/'best.qr',provenance='untrained',max_gap=args.max_gap)
        (output/'deployment.json').write_text(json.dumps({'accepted':False,'selected_epoch':0,
            'reason':'No candidate accepted; best.qr is an untrained identity baseline.'},indent=2)+'\n')
    for epoch in range(args.epochs):
        args.round_index=epoch
        stage=('r' if epoch%2==0 else 'q') if args.mode=='qr' else args.mode
        for name in ('q','r'):
            for parameter in getattr(model,name).parameters():parameter.requires_grad_(name==stage)
        result=dataset_pass(model,routes['train'],args,optimizer=optimizers[stage])
        validation=dataset_pass(model,routes['validation'],args) if routes['validation'] else None
        score=(validation or result)['selection_weighted_rmse_m']
        checkpoint_save(output/'last.pt',model,max_gap=args.max_gap,epoch=epoch+1,mode=args.mode)
        checkpoint_save(output/f'epoch{epoch+1:02d}.pt',model,max_gap=args.max_gap,epoch=epoch+1,mode=args.mode)
        model.export(output/'last.qr',provenance=args.provenance,max_gap=args.max_gap)
        model.export(output/f'epoch{epoch+1:02d}.qr',provenance=args.provenance,max_gap=args.max_gap)
        decision=None;native=None
        if safe:
            native=native_validation(model,routes['validation'],args,output/'candidate.qr')
            decision=deployment_decision(baseline,native,min_improvement=args.selection_min_improvement)
            score=native['aggregate']['rms_3d_m']
        if score is not None and score<best and (not safe or decision['accepted']):
            best=score;selected_epoch=epoch+1
            checkpoint_save(output/'best.pt',model,max_gap=args.max_gap,epoch=epoch+1,mode=args.mode)
            model.export(output/'best.qr',provenance=args.provenance,max_gap=args.max_gap)
            (output/'deployment.json').write_text(json.dumps({'accepted':bool(safe),
                'selected_epoch':selected_epoch,'decision':decision,
                'note':'Validation evidence only, not a general guarantee.'},indent=2,allow_nan=False)+'\n')
        row={'epoch':epoch+1,'stage':stage,'train':result,'validation':validation,
             'native_validation':native,'deployment_decision':decision,
             'best_selection_rmse_m':best,'selected_epoch':selected_epoch}
        with (output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(f"epoch={epoch+1} stage={stage} supervised={result['matched_epochs']} steps={result['optimizer_steps']} selection_rmse={score:.6f}m",flush=True)
    if safe and selected_epoch==0:
        print('NO_CANDIDATE_ACCEPTED: best.qr is untrained identity, not a learned improvement.',flush=True)
    if not safe:
        print('FLOAT/debug selection only: native RTK acceptance has not been established.',flush=True)
    return output
