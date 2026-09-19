"""Train, export, replay and evaluate learned Q/R with local user data."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import math
import os
import tempfile
from pathlib import Path
import sys
import numpy as np
from .native import Session
from .reference import Reference,enu_rotation,summarize
from .training import train,load_manifest,checkpoint_load

def evaluate_records(records,reference,*,fix_threshold=.1):
    errors=[];statuses=[];available=fixed=refvalid=0
    for record in records:
        position=np.asarray(record['position']);status=int(record['status'])
        available+=int(status>0 and np.isfinite(position).all());fixed+=int(status==1)
        target,valid=reference.at(float(record['time']));refvalid+=int(valid)
        if valid and status>0 and np.isfinite(position).all():
            errors.append(enu_rotation(target)@(position-target));statuses.append(status)
    result=summarize(errors,statuses,attempted=len(records),fix_threshold=fix_threshold)
    result.update(solution_available_epochs=available,solution_fixed_epochs=fixed,reference_available_epochs=refvalid,
                  solution_fixed_fraction=fixed/len(records) if records else 0.)
    return result

def replay(args):
    if args.legacy and (args.mode!='off' or args.model):raise ValueError('--legacy requires --mode off and no model')
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError(f'output already exists: {output}')
    output.parent.mkdir(parents=True,exist_ok=True);records=[]
    with Session(config=args.config,rover=args.rover,base=args.base,nav=args.nav,model=args.model,mode=args.mode,
                 training=args.float,allow_test=args.allow_test_model,max_gap=args.max_gap,pair_age=args.pair_age,
                 library=args.library,stable=not args.legacy) as session:
        while not args.max_epochs or len(records)<args.max_epochs:
            record=session.step()
            if record is None:break
            records.append(record)
    if not records:raise ValueError('no rover epochs')
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['t_gpst_s','x_m','y_m','z_m','status','ns','ratio','age_s'])
        for row in records:writer.writerow([format(row['time'],'.9f'),*row['position'],row['status'],row['ns'],row['ratio'],row['age']])
    if args.reference:
        reference=Reference.load(args.reference,max_gap=args.reference_max_gap,time_offset=args.reference_time_offset)
        result=evaluate_records(records,reference,fix_threshold=args.fix_threshold)
        result['solver']='legacy' if args.legacy else 'stable';result['learned_mode']=args.mode
        output.with_suffix('.metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2,allow_nan=False))
    print(f'Wrote {len(records)} epochs to {output}')

def evaluate(args):
    reference=Reference.load(args.reference,max_gap=args.reference_max_gap,time_offset=args.reference_time_offset)
    records=[]
    with Path(args.solution).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            records.append({'time':float(row['t_gpst_s']),'position':[float(row[c]) for c in ('x_m','y_m','z_m')],'status':int(row['status'])})
    result=evaluate_records(records,reference,fix_threshold=args.fix_threshold)
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))

def convert_reference(args):
    """Convert an independent RTKLIB truth .pos with explicitly declared formats."""
    quality={int(v) for v in args.quality.split(',')};rows=[]
    for number,line in enumerate(Path(args.input).read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        if line.lstrip().startswith('%'):
            if 'UTC' in line.upper():raise ValueError('UTC input is not GPST; convert its time scale explicitly first')
            continue
        fields=line.split()
        if len(fields)<6:raise ValueError(f'line {number}: expected RTKLIB time, coordinates and quality')
        if args.time_format=='week-tow':
            week=int(fields[0]);tow=float(fields[1])
        else:
            stamp=dt.datetime.strptime(fields[0]+' '+fields[1],'%Y/%m/%d %H:%M:%S.%f') if '.' in fields[1] else dt.datetime.strptime(fields[0]+' '+fields[1],'%Y/%m/%d %H:%M:%S')
            seconds=(stamp-dt.datetime(1980,1,6)).total_seconds();week=int(seconds//604800);tow=seconds-week*604800
        xyz=np.array([float(v) for v in fields[2:5]])
        if args.coordinate_format=='llh':
            lat,lon=map(math.radians,xyz[:2]);h=xyz[2];e2=6.6943799901413165e-3
            N=6378137/math.sqrt(1-e2*math.sin(lat)**2)
            xyz=np.array([(N+h)*math.cos(lat)*math.cos(lon),(N+h)*math.cos(lat)*math.sin(lon),(N*(1-e2)+h)*math.sin(lat)])
        rows.append([week,format(tow,'.9f'),*xyz,int(int(fields[5]) in quality)])
    p=Path(args.output)
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.reference-',suffix='.csv',dir=p.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['gps_week','tow_s','x_m','y_m','z_m','valid']);writer.writerows(rows)
        Reference.load(temporary) # validate before publishing; never replace an existing file
        os.link(temporary,p)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Converted {len(rows)} reference rows; LLH heights must be ellipsoidal.')

def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train',help='reference-supervised live native FLOAT training')
    t.add_argument('--manifest',required=True);t.add_argument('--config',required=True);t.add_argument('--output',required=True)
    t.add_argument('--epochs',type=int,default=20);t.add_argument('--sequence-length',type=int,default=16)
    t.add_argument('--warmup-epochs',type=int,default=5);t.add_argument('--max-epochs-per-route',type=int,default=0)
    t.add_argument('--mode',choices=['q','r','qr'],default='qr');t.add_argument('--learning-rate',type=float,default=1e-3)
    t.add_argument('--gradient-clip',type=float,default=1.);t.add_argument('--up-weight',type=float,default=1.)
    t.add_argument('--huber-delta',type=float,default=1.,help='metres; zero selects squared loss')
    t.add_argument('--nll-weight',type=float,default=1e-3);t.add_argument('--reg-weight',type=float,default=1e-4)
    t.add_argument('--q-limit',type=float,default=10.);t.add_argument('--r-limit',type=float,default=1000.)
    t.add_argument('--device',default='cpu');t.add_argument('--threads',type=int,default=1);t.add_argument('--seed',type=int,default=7)
    t.add_argument('--init',help='safe weights-only checkpoint initialization (not optimizer resume)')
    t.add_argument('--allow-training-only',action='store_true');t.add_argument('--provenance',choices=['trained','synthetic'],default='trained')
    t.set_defaults(handler=train)
    r=sub.add_parser('replay',help='native C replay with original integer fixing')
    r.add_argument('--config',required=True);r.add_argument('--rover',required=True);r.add_argument('--base',required=True)
    r.add_argument('--nav',nargs='+',required=True);r.add_argument('--model');r.add_argument('--mode',choices=['off','q','r','qr'],default='off')
    r.add_argument('--float',action='store_true',help='disable AR for FLOAT-only comparison')
    r.add_argument('--legacy',action='store_true',help='unmodified legacy covariance update; mode off only')
    r.add_argument('--allow-test-model',action='store_true');r.add_argument('--max-epochs',type=int,default=0)
    r.add_argument('--output',required=True);r.add_argument('--reference');r.set_defaults(handler=replay)
    e=sub.add_parser('evaluate',help='evaluate replay CSV against independent reference')
    e.add_argument('--solution',required=True);e.add_argument('--reference',required=True);e.add_argument('--output',required=True);e.set_defaults(handler=evaluate)
    c=sub.add_parser('convert-reference',help='convert independent truth .pos to explicit GPST/ECEF CSV')
    c.add_argument('--input',required=True);c.add_argument('--output',required=True)
    c.add_argument('--coordinate-format',choices=['xyz','llh'],required=True)
    c.add_argument('--time-format',choices=['week-tow','calendar-gpst'],required=True)
    c.add_argument('--quality',default='1',help='comma separated trusted reference quality codes')
    c.set_defaults(handler=convert_reference)
    x=sub.add_parser('export',help='export a safe tensor checkpoint for C-only deployment')
    x.add_argument('--checkpoint',required=True);x.add_argument('--output',required=True)
    x.add_argument('--provenance',choices=['trained','synthetic'],required=True)
    def export(args):
        if Path(args.output).exists():raise FileExistsError(args.output)
        model,meta=checkpoint_load(args.checkpoint);model.export(args.output,provenance=args.provenance,max_gap=meta['max_gap'])
    x.set_defaults(handler=export)
    for command in (t,r):
        command.add_argument('--library',help='explicit rebuilt shared RTKLIB path')
        command.add_argument('--max-gap',type=float,default=30.,help='seconds; must match exported model')
        command.add_argument('--pair-age',type=float,default=.05,help='maximum rover/base timestamp difference')
    for command in (t,r,e):command.add_argument('--reference-max-gap',type=float,default=1.,help='maximum interpolation gap, seconds')
    for command in (r,e):
        command.add_argument('--reference-time-offset',type=float,default=0.,help='seconds ADDED to reference GPST')
        command.add_argument('--fix-threshold',type=float,default=.1,help='metres; fixed large-position-error proxy, not integer truth')
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    try:args.handler(args);return 0
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,FloatingPointError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
