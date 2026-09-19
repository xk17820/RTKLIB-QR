"""Live native RINEX tests. Header coordinates are test targets, NOT truth."""
from pathlib import Path
import ctypes
import importlib
import numpy as np
import pytest
import torch
from qrlearn.model import QRModel

ROOT=Path(__file__).resolve().parents[3]
KIT=Path(__file__).resolve().parents[1]
DATA=ROOT/'test/data/rinex'

def test_library_exports_live_bridge():
    lib=ctypes.CDLL(str(ROOT/'lib/librtklib.so'))
    assert hasattr(lib,'qr_open'), 'native RINEX training bridge not implemented'
    assert lib.qr_bridge_abi()==1

def modules():
    assert (KIT/'qrlearn/native.py').exists(), 'native Python session not implemented'
    assert (KIT/'qrlearn/live.py').exists(), 'differentiable live tape not implemented'
    return importlib.import_module('qrlearn.native'), importlib.import_module('qrlearn.live')

def session_kwargs():
    return dict(config=KIT/'configs/short-baseline.conf',rover=DATA/'30400920.05o',
                base=DATA/'07590920.05o',nav=[DATA/'30400920.05n',DATA/'07590920.05n'],
                max_gap=31.0,pair_age=0.05)

def test_raw_rinex_loss_reaches_both_branches():
    native,live=modules()
    torch.manual_seed(3)
    model=QRModel().double();engine=live.TraceEngine(model)
    loss=[];records=[]
    # Deliberately artificial reference only checks gradients, not accuracy.
    target=torch.tensor([-3978242.3348,3382841.0715,3649902.9167],dtype=torch.float64)
    with native.Session(**session_kwargs(),training=True,mode='qr') as s:
        for _ in range(12):
            record=s.step(engine.handle);records.append(record)
            if record['status']==2:loss.append((engine.position-target).square().mean())
    assert len(loss)>3
    torch.stack(loss).mean().backward()
    for branch in (model.q,model.r):
        g=sum(p.grad.abs().sum().item() for p in branch.parameters() if p.grad is not None)
        assert np.isfinite(g) and g>1e-12
    assert engine.max_state_discrepancy<0.01
    assert engine.updates>3

def test_native_model_and_callback_forward_agree(tmp_path):
    native,live=modules()
    model=QRModel().double()
    with torch.no_grad():
        model.q.fc2.bias.fill_(.04);model.r.fc2.bias.fill_(.025)
    file=tmp_path/'synthetic.qr';model.export(file,provenance='synthetic',max_gap=31)
    engine=live.TraceEngine(model);a=[];b=[]
    with native.Session(**session_kwargs(),training=True,mode='qr') as s:
        for _ in range(10):a.append(s.step(engine.handle)['position'])
    with native.Session(**session_kwargs(),training=True,mode='qr',model=file,allow_test=True) as s:
        for _ in range(10):b.append(s.step()['position'])
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-5)

def test_restart_is_reproducible_and_callback_error_aborts():
    native,_=modules()
    with native.Session(**session_kwargs(),training=True,mode='off') as s:
        a=[s.step()['position'] for _ in range(4)]
        s.restart()
        b=[s.step()['position'] for _ in range(4)]
        np.testing.assert_array_equal(a,b)
        def bad(event):raise ValueError('intentional callback failure')
        with pytest.raises(ValueError,match='intentional callback failure'):s.step(bad)

def test_unsupported_training_config_is_rejected(tmp_path):
    native,_=modules();p=tmp_path/'bad.conf'
    p.write_text('pos1-posmode=kinematic\npos1-dynamics=off\n')
    args=session_kwargs();args['config']=p
    with pytest.raises(RuntimeError,match='dynamics'):native.Session(**args,training=True)

def test_untrained_model_is_rejected_without_opt_in(tmp_path):
    native,_=modules();p=tmp_path/'bad.qr';QRModel().export(p,provenance='untrained')
    with pytest.raises(RuntimeError,match='model'):native.Session(**session_kwargs(),model=p,mode='qr')

def test_live_covariance_stays_symmetric_positive_on_30s_data():
    native,live=modules();engine=live.TraceEngine(QRModel().double())
    with native.Session(**session_kwargs(),training=True,mode='qr') as s:
        for _ in range(10):
            s.step(engine.handle)
            P=engine.P.detach().numpy();idx=np.flatnonzero(np.diag(P)>0)
            np.testing.assert_allclose(P,P.T,rtol=0,atol=1e-7)
            assert np.linalg.eigvalsh(P[np.ix_(idx,idx)]).min()>-1e-7


def test_fixed_baseline_constraint_rejected_before_training(tmp_path):
    native,_=modules();config=tmp_path/'constraint.conf'
    config.write_text((KIT/'configs/short-baseline.conf').read_text()+'\npos2-baselen=3\n')
    args=session_kwargs();args['config']=config
    with pytest.raises(RuntimeError,match='baseline constraint'):
        native.Session(**args,training=True)


def test_c_only_replay_matches_native_session(tmp_path):
    import subprocess,os,csv
    native,_=modules()
    source=KIT/'examples/replay_native.c'
    assert source.exists(), 'C-only bridge replay example is missing'
    executable=tmp_path/'qrreplay';output=tmp_path/'out.csv'
    subprocess.run(['cc','-std=c99','-I'+str(ROOT/'src'),str(source),'-L'+str(ROOT/'lib'),
        '-Wl,-rpath,'+str(ROOT/'lib'),'-lrtklib','-lm','-o',str(executable)],check=True,capture_output=True)
    args=session_kwargs();env={k:v for k,v in os.environ.items() if not k.startswith('RTKLIB_QR')}
    subprocess.run([str(executable),str(args['config']),str(args['rover']),str(args['base']),'-',str(output),
                   '31',*[str(p) for p in args['nav']]],env=env,check=True,capture_output=True)
    with output.open() as f:records=list(csv.DictReader(f))
    assert len(records)>20
    with native.Session(**args,training=False,mode='off') as session:
        for record in records[:12]:
            expected=session.step()
            assert int(record['status'])==expected['status']
            np.testing.assert_allclose([float(record[c]) for c in ('x_m','y_m','z_m')],expected['position'],atol=1e-8,rtol=0)
