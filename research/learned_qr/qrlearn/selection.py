"""Native complete-RTK model selection. References never enter inference."""
from __future__ import annotations
import math
import numpy as np
from .native import Session
from .reference import Reference,enu_rotation,summarize


def native_validation(model,routes,args,path,*,baseline=False):
    model.export(path,provenance='untrained' if baseline else args.provenance,max_gap=args.max_gap)
    all_e=[];all_s=[];total=0;stats={}
    for route in routes:
        ref=Reference.load(route['reference'],max_gap=float(route.get('reference_max_gap_s',args.reference_max_gap)),
            time_offset=float(route.get('reference_time_offset_s',0)))
        errors=[];statuses=[];matches=[];available=[];n=fixed=0
        with Session(config=route.get('config',args.config),rover=route['rover'],base=route['base'],
            nav=route['nav'],model=None if baseline else path,mode='off' if baseline else args.mode,
            training=False,allow_test=True,max_gap=args.max_gap,pair_age=args.pair_age,library=args.library) as sess:
            while (r:=sess.step()) is not None:
                n+=1;fixed+=int(r['status']==1)
                ok=r['status']>0 and np.isfinite(r['position']).all()
                if ok:available.append(r['time'])
                truth,valid=ref.at(r['time'])
                if ok and valid:
                    errors.append(enu_rotation(truth)@(r['position']-truth))
                    statuses.append(r['status']);matches.append(r['time'])
        a=summarize(errors,statuses,attempted=n,fix_threshold=args.selection_fix_threshold)
        a.update(solution_available_epochs=len(available),solution_fixed_fraction=fixed/n if n else 0.,
                 matched_times=matches,available_times=available)
        stats[route['id']]=a;total+=n;all_e.extend(errors);all_s.extend(statuses)
    return dict(aggregate=summarize(all_e,all_s,attempted=total,fix_threshold=args.selection_fix_threshold),routes=stats)


def deployment_decision(baseline,candidate,*,min_improvement=.01):
    if not math.isfinite(min_improvement) or not 0<=min_improvement<1:raise ValueError('invalid minimum improvement')
    reasons=[];gain=None;b=baseline.get('aggregate',{}).get('rms_3d_m');v=candidate.get('aggregate',{}).get('rms_3d_m')
    old_routes=baseline.get('routes',{});new_routes=candidate.get('routes',{})
    if not old_routes or set(old_routes)!=set(new_routes):reasons.append('validation routes missing/changed')
    if b is None or v is None or not math.isfinite(b) or not math.isfinite(v) or b<=0:
        reasons.append('no finite positive baseline/candidate RMS')
    else:
        gain=(b-v)/b
        if gain<=0 or gain<min_improvement:reasons.append('insufficient gain over baseline')
    for key in sorted(set(old_routes)&set(new_routes)):
        old,new=old_routes[key],new_routes[key]
        for field in ('attempted_epochs','matched_epochs','solution_available_epochs','matched_times','available_times'):
            if old.get(field)!=new.get(field):reasons.append(f'{key}: changed {field}')
        if not old.get('matched_epochs') or not new.get('matched_epochs'):
            reasons.append(f'{key}: zero matched reference');continue
        for field,ratio,absolute in [('rms_3d_m',1.02,.001),('p95_3d_m',1.05,.005)]:
            x,y=new.get(field),old.get(field)
            if x is None or y is None or not math.isfinite(x) or not math.isfinite(y) or x>ratio*y+absolute:
                reasons.append(f'{key}: {field} regression')
        if new['solution_fixed_fraction']+.05<old['solution_fixed_fraction']:reasons.append(f'{key}: FIX fraction drop >5pp')
        if new['fixed_large_error_proxy_count']>old['fixed_large_error_proxy_count']:reasons.append(f'{key}: increased fixed-position-error proxy')
    return dict(accepted=not reasons,reasons=reasons,relative_improvement=gain,baseline_rms_3d_m=b,candidate_rms_3d_m=v)
