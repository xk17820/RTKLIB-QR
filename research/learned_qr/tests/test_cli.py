from pathlib import Path
import importlib
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from qrlearn.native import Session
from tests_support import raw_kwargs
KIT=Path(__file__).resolve().parents[1]

def cli_module():
    assert (KIT/'qrlearn/cli.py').exists(), 'end-user training/replay commands not implemented'
    return importlib.import_module('qrlearn.cli')

def test_manifest_rejects_same_rover_for_training_and_validation(tmp_path):
    m=cli_module();route=dict(id='one',rover='/tmp/a',base='/tmp/b',nav=['/tmp/n'],reference='/tmp/ref')
    p=tmp_path/'m.json';p.write_text(json.dumps({'train':[route],'validation':[dict(route,id='two')]}))
    with pytest.raises(ValueError,match='overlap'):m.load_manifest(p,allow_training_only=False)

def test_cli_train_export_and_native_replay(tmp_path):
    cli_module();args=raw_kwargs();ref=tmp_path/'test_reference.csv'
    lines=['gps_week,tow_s,x_m,y_m,z_m,valid']
    # Artificial offsets to native solutions are a plumbing test, not ground truth.
    with Session(**args,training=True,mode='off') as s:
        for _ in range(8):
            row=s.step();week=int(row['time']//604800);tow=row['time']-week*604800
            xyz=np.array(row['position'])+[.1,-.1,.15]
            lines.append(f'{week},{tow:.9f},{xyz[0]:.9f},{xyz[1]:.9f},{xyz[2]:.9f},1')
    ref.write_text('\n'.join(lines)+'\n')
    manifest=tmp_path/'m.json';manifest.write_text(json.dumps({'train':[dict(id='fixture',
        rover=str(args['rover']),base=str(args['base']),nav=[str(p) for p in args['nav']],reference=str(ref))]}))
    out=tmp_path/'trained'
    command=[sys.executable,'-m','qrlearn.cli','train','--manifest',str(manifest),'--config',str(args['config']),
        '--output',str(out),'--r-policy','unconstrained','--selection','float','--epochs','2','--sequence-length','4','--warmup-epochs','1',
        '--max-epochs-per-route','8','--max-gap','31','--allow-training-only','--provenance','synthetic']
    subprocess.run(command,cwd=KIT,check=True,capture_output=True,text=True)
    assert (out/'best.qr').exists() and (out/'history.jsonl').exists()
    checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
    for branch in ('q','r'):assert torch.count_nonzero(checkpoint['state_dict'][branch+'.fc2.bias'])>0
    predictions=tmp_path/'replay.csv'
    cmd=[sys.executable,'-m','qrlearn.cli','replay','--config',str(args['config']),
         '--rover',str(args['rover']),'--base',str(args['base']),'--nav',*[str(p) for p in args['nav']],
         '--model',str(out/'best.qr'),'--mode','qr','--allow-test-model','--max-gap','31',
         '--max-epochs','8','--output',str(predictions),'--reference',str(ref)]
    subprocess.run(cmd,cwd=KIT,check=True,capture_output=True,text=True)
    metrics=json.loads(predictions.with_suffix('.metrics.json').read_text())
    assert metrics['matched_epochs']>0
    assert metrics['attempted_epochs']==8

def test_no_reference_overlap_fails_instead_of_exporting_model(tmp_path):
    m=cli_module();args=raw_kwargs()
    ref=tmp_path/'ref.csv';ref.write_text('gps_week,tow_s,x_m,y_m,z_m,valid\n2400,0,6378137,0,0,1\n')
    manifest=tmp_path/'m.json';manifest.write_text(json.dumps({'train':[dict(id='bad-time',
        rover=str(args['rover']),base=str(args['base']),nav=[str(p) for p in args['nav']],reference=str(ref))]}))
    out=tmp_path/'out'
    rc=m.main(['train','--manifest',str(manifest),'--config',str(args['config']),'--output',str(out),
       '--epochs','1','--max-epochs-per-route','4','--warmup-epochs','0','--max-gap','31','--allow-training-only'])
    assert rc!=0 and not (out/'best.qr').exists()


def test_reference_conversion_does_not_leave_invalid_output(tmp_path):
    m=cli_module();source=tmp_path/'truth.pos';output=tmp_path/'truth.csv'
    source.write_text('2300 1 6378137 0 0 1\n2300 1 6378137 0 0 1\n')
    rc=m.main(['convert-reference','--input',str(source),'--output',str(output),
               '--coordinate-format','xyz','--time-format','week-tow'])
    assert rc==2
    assert not output.exists(), 'failed conversion must not leave a reference CSV'


def test_rejected_native_candidate_keeps_identity_checkpoint(tmp_path):
    """Solver-derived labels are ONLY a software rejection test, not truth."""
    import shutil
    m=cli_module();kw=raw_kwargs();copy=tmp_path/'validation.obs'
    shutil.copyfile(kw['rover'],copy);records=[]
    with Session(**kw,training=False,mode='off') as sess:
        for _ in range(8):records.append(sess.step())
    def reference(path,offset):
        rows=['t_gpst_s,x_m,y_m,z_m,valid']
        for rec in records:rows.append(','.join(map(str,[rec['time'],*(rec['position']+offset),1])))
        path.write_text('\n'.join(rows)+'\n')
    tr=tmp_path/'tr.csv';va=tmp_path/'va.csv';reference(tr,.2);reference(va,0.)
    def route(name,obs,ref):return dict(id=name,rover=str(obs),base=str(kw['base']),
                                        nav=list(map(str,kw['nav'])),reference=str(ref))
    manifest=tmp_path/'m.json';manifest.write_text(json.dumps({'train':[route('tr',kw['rover'],tr)],
                                                             'validation':[route('va',copy,va)]}))
    out=tmp_path/'out'
    rc=m.main(['train','--manifest',str(manifest),'--config',str(kw['config']),'--output',str(out),
        '--epochs','1','--sequence-length','4','--warmup-epochs','1','--max-epochs-per-route','8',
        '--max-gap','31','--provenance','synthetic'])
    assert rc==0
    assert json.loads((out/'deployment.json').read_text())['accepted'] is False
    assert torch.load(out/'best.pt',map_location='cpu',weights_only=True)['epoch']==0
    assert (out/'best.qr').read_text().startswith('RTKLIB_QR_V2 0 ')

    assert json.loads((out/"run.json").read_text())["selection"]=="native complete-RTK gate against identity baseline"
