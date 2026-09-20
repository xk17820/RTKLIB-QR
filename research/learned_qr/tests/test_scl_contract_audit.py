"""Full bundled RINEX contract checks; header coordinates are NOT truth."""
import ctypes as C
import hashlib
from pathlib import Path
import re
import numpy as np
import pytest
import torch
from qrlearn.model import QRModel
from qrlearn.native import D, Session, array
from qrlearn.scl import condition_position
from tests_support import raw_kwargs


def record_tuple(rec):
    return (rec['time'],rec['status'],rec['ns'],rec['ratio'],rec['age'],
            tuple(rec['position']))


def collect(kwargs, *, training, candidates, audit_rows=None):
    records=[]; states=[]; accepted=[]; proposals=[]
    def handler(e):
        if e.kind==14:
            states.append(hashlib.sha256(array(e.x,e.nx).tobytes()+
                                         array(e.P,e.nx*e.nx).tobytes()).hexdigest())
        elif e.kind==16:
            accepted.append((e.time,array(e.ix,2*e.n).tolist(),array(e.a,e.n).tolist()))
        elif e.kind==15:
            pairs=array(e.ix,2*e.n).reshape(e.n,2)
            integers=array(e.a,e.n)
            assert pairs.min()>=3 and pairs.max()<e.nx
            x=array(e.x,e.nx)
            P=array(e.P,e.nx*e.nx).reshape(e.nx,e.nx,order='F')
            a,b=pairs.T
            Q=P[np.ix_(a,a)]-P[np.ix_(a,b)]-P[np.ix_(b,a)]+P[np.ix_(b,b)]
            np.linalg.cholesky((Q+Q.T)*.5)
            # Recompute using the UNMODIFIED native integer solver. Its final
            # back-transformation is a floating solve Z'\E, not integer dtype:
            # large initial ambiguities can have microcycle rounding residue.
            # Preserve those exact values rather than silently rounding them.
            y=np.ascontiguousarray(x[a]-x[b]);native_Q=np.asfortranarray(Q)
            native_ints=np.zeros((e.n,2),order='F');native_cost=np.zeros(2)
            pointers=[z.ctypes.data_as(D) for z in (y,native_Q,native_ints,native_cost)]
            assert native_lambda(e.n,2,*pointers)==0
            np.testing.assert_array_equal(integers,native_ints[:,0])
            np.testing.assert_array_equal(array(e.c,2),native_cost)
            pos=condition_position(torch.tensor(x),torch.tensor(P),
                                   torch.tensor(pairs),torch.tensor(integers)).numpy()
            np.testing.assert_allclose(pos,array(e.b,3),atol=1e-5,rtol=0)
            if audit_rows is not None:
                audit_rows.append(dict(time_gpst_s=e.time,dd_pairs=pairs.tolist(),
                    ambiguity_unit='cycles',Qaa_unit='cycles^2',Qpa_unit='m*cycles',
                    float_dd_cycles=(x[a]-x[b]).tolist(),candidate_cycles=integers.tolist(),
                    Qaa=Q.tolist(),Qpa=(P[:3,a]-P[:3,b]).tolist(),
                    float_ecef_m=x[:3].tolist(),conditional_native_ecef_m=array(e.b,3).tolist(),
                    conditional_torch_ecef_m=pos.tolist(),candidate_costs=native_cost.tolist()))
            proposals.append(dict(time=e.time,dd_count=e.n,
                integer_roundoff_cycles=float(np.abs(integers-np.rint(integers)).max()),
                conditional_discrepancy_m=float(np.abs(pos-array(e.b,3)).max())))
    with Session(**kwargs,training=training,mode='off',candidates=candidates,
                 observe_fixed=not training) as session:
        native_lambda=getattr(session.lib,'lambda')
        native_lambda.argtypes=[C.c_int,C.c_int,D,D,D,D]
        native_lambda.restype=C.c_int
        while (rec:=session.step(handler)) is not None:
            records.append(record_tuple(rec))
    return records,states,accepted,proposals


@pytest.mark.parametrize('armode',['continuous','fix-and-hold'])
def test_candidate_observer_is_read_only_through_complete_native_rtk_sequence(tmp_path,armode,capfd):
    kwargs=raw_kwargs(); config=tmp_path/'mode.conf'
    config.write_text(kwargs['config'].read_text()+'\npos2-armode ='+armode+'\n')
    kwargs['config']=config
    on=collect(kwargs,training=False,candidates=True)
    off=collect(kwargs,training=False,candidates=False)
    assert len(on[0])==120  # full bundled fixture, no max-epoch truncation
    assert on[:3]==off[:3], 'observer must not change outputs, float state/P or accepted integers'
    assert len(on[3])>len(on[2])>0  # candidates not limited to accepted FIX
    assert not off[3]
    assert "invalid option value" not in capfd.readouterr().err


def test_candidates_are_available_on_continuous_training_without_accepted_fix():
    recs,_,accepted,proposals=collect(raw_kwargs(),training=True,candidates=True)
    assert len(recs)==120 and len(proposals)>100
    assert all(row[1]!=1 for row in recs)
    assert not accepted


def rinex_prefix(source,destination,count):
    lines=Path(source).read_text().splitlines(keepends=True)
    pattern=re.compile(r'^ \d{2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\.\d+\s+[0-6]\s*\d+')
    starts=[i for i,line in enumerate(lines) if pattern.match(line)]
    assert len(starts)>count
    destination.write_text(''.join(lines[:starts[count]]))
    return destination


def inference_prefix(kwargs,model,limit):
    features=[]; records=[]
    def handler(e):
        if e.kind in (5,6):
            shape=(10,e.n+1);h=array(e.a,shape[0]*shape[1]).reshape(shape)
            features.append((e.time,e.kind,tuple(array(e.ix,2)) if e.kind==6 else (),h.copy()))
            net=model.q if e.kind==5 else model.r
            with torch.no_grad(): values=net(torch.tensor(h,dtype=torch.float64)).numpy()
            for i,v in enumerate(values): e.b[i]=v
    with Session(**kwargs,mode='qr',training=True,scl=True) as session:
        for _ in range(limit):
            rec=session.step(handler)
            assert rec is not None
            records.append(record_tuple(rec))
    return features,records


def test_current_features_are_invariant_to_unavailable_future_observation_suffix(tmp_path):
    kwargs=raw_kwargs(); short=dict(kwargs);n=20
    for key in ('rover','base'):
        short[key]=rinex_prefix(kwargs[key],tmp_path/(key+'.obs'),n)
    torch.manual_seed(77);model=QRModel(r_policy='scl').double()
    with torch.no_grad():
        model.q.fc2.bias.fill_(.01);model.r.fc2.bias.fill_(.1)
    full=inference_prefix(kwargs,model,n);prefix=inference_prefix(short,model,n)
    assert full[1]==prefix[1]
    assert len(full[0])==len(prefix[0])>n
    for a,b in zip(full[0],prefix[0]):
        assert a[:3]==b[:3]
        np.testing.assert_array_equal(a[3],b[3])


def test_rover_rinex_approximate_header_is_not_a_reference_initialization(tmp_path):
    kwargs=raw_kwargs(); changed=dict(kwargs);p=tmp_path/'changed.obs'
    text=kwargs['rover'].read_text()
    lines=text.splitlines(keepends=True)
    for i,line in enumerate(lines):
        if 'APPROX POSITION XYZ' in line:
            lines[i]=f'{1000000.:14.4f}{2000000.:14.4f}{3000000.:14.4f}'+line[42:]
            break
    else:raise AssertionError('fixture has no approximate header')
    p.write_text(''.join(lines));changed['rover']=p
    model=QRModel(r_policy='scl').double()
    a=inference_prefix(kwargs,model,20);b=inference_prefix(changed,model,20)
    assert a[1]==b[1]
    for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x[3],y[3])


def inject_fixture_code_step(source,destination):
    """Only this four-observable RINEX2 fixture; software fault, NOT field data."""
    text=Path(source).read_text();lines=text.splitlines(keepends=True)
    assert '     4    L1    C1    L2    P2' in text
    epoch=-1;changed=0;i=next(i for i,v in enumerate(lines) if 'END OF HEADER' in v)+1
    while i<len(lines):
        header=lines[i];count=int(header[29:32]);flag=int(header[28:29])
        if flag==4:  # retained RINEX header-update event, not a rover epoch
            i+=count+1;continue
        assert count<=12 and flag in (0,1)
        epoch+=1
        satellites=[header[32+j*3:35+j*3].strip() for j in range(count)]
        for j,sat in enumerate(satellites):
            if sat=='G11' and 20<=epoch<80:
                k=i+j+1;line=lines[k]
                lines[k]=line[:16]+f'{float(line[16:30])+24.:14.3f}'+line[30:]
                changed+=1
        i+=count+1
    assert epoch+1==120 and changed>40
    destination.write_text(''.join(lines));return destination


@pytest.mark.parametrize('fault',[False,True])
def test_full_native_scl_tape_keeps_conditional_gradients_across_tbptt_windows(tmp_path,fault):
    """Arbitrary scalar objective is a derivative probe, NOT reference accuracy."""
    from qrlearn.scl import CompactTrace
    torch.manual_seed(37)
    model=QRModel(r_policy='scl').double()
    engine=CompactTrace(model)
    kwargs=raw_kwargs()
    if fault:kwargs['rover']=inject_fixture_code_step(kwargs['rover'],tmp_path/'code_step.obs')
    losses=[];gradient_totals=np.zeros(2);epochs=windows=0;open_gate_rows=[]
    def handle(event):
        if event.kind==6:
            h=array(event.a,10*(event.n+1)).reshape(10,event.n+1)
            if h[-1,3]>.5 and h[-1,-1]>.5:
                evidence=np.expm1(abs(h[-1,8:10])).max()
                if evidence>model.r.anchor:open_gate_rows.append(event.time)
        engine.handle(event)
    def flush():
        nonlocal windows
        if losses:
            model.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            for i,net in enumerate((model.q,model.r)):
                values=[p.grad for p in net.parameters() if p.grad is not None]
                assert all(torch.isfinite(v).all() for v in values)
                gradient_totals[i]+=sum(v.abs().sum().item() for v in values)
            windows+=1
        engine.detach();losses.clear()
    with Session(**kwargs,training=True,mode='qr',scl=True,candidates=True) as session:
        while (rec:=session.step(handle)) is not None:
            epochs+=1
            if engine.fixed_position is not None:
                # No purported ground truth is used, nor is an optimizer run.
                losses.append((engine.position.square().sum()+engine.fixed_position.square().sum())*1e-7)
            if epochs%16==0:flush()
        flush()
    assert epochs==120 and windows==8 and engine.candidate_count==120
    assert np.all(np.isfinite(gradient_totals)) and gradient_totals[0]>0
    if fault:
        assert open_gate_rows and gradient_totals[1]>0
    else:
        assert not open_gate_rows and gradient_totals[1]==0
    assert engine.max_state_discrepancy<.01
    assert engine.max_candidate_discrepancy<.01
