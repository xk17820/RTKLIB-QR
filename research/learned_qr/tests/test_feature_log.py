"""Read-only native feature logging must not perturb positioning or original AR."""
import csv
import numpy as np
import pytest
from qrlearn.native import Session
from tests_support import raw_kwargs


def test_native_feature_log_is_read_only_and_complete(tmp_path):
    results=[]
    for log in (None,tmp_path/'features.csv'):
        rows=[]
        with Session(**raw_kwargs(),mode='off',training=False,scl=True,feature_log=log) as s:
            while (r:=s.step()) is not None:rows.append(r)
        results.append(rows)
    assert len(results[0])==len(results[1])==120
    for a,b in zip(*results):
        assert a['status']==b['status'] and a['ratio']==b['ratio']
        np.testing.assert_array_equal(a['position'],b['position'])
    with (tmp_path/'features.csv').open() as f:rows=list(csv.DictReader(f))
    assert set(r['kind'] for r in rows)=={'Q','R'}
    assert all(float(r['scale0'])==1. for r in rows)
    times={float(r['time']) if 'time' in r else float(r['t_gpst_s']) for r in rows if r['kind']=='R'}
    assert len(times)==120


def test_logging_refuses_existing_file(tmp_path):
    p=tmp_path/'existing.csv';p.write_text('preserve me')
    with pytest.raises(FileExistsError):Session(**raw_kwargs(),feature_log=p)
    assert p.read_text()=='preserve me'


def test_native_feature_log_survives_filter_gap_resets(tmp_path):
    opts=raw_kwargs();opts['max_gap']=20.
    p=tmp_path/'gaps.csv'
    with Session(**opts,mode='off',training=False,scl=True,feature_log=p) as s:
        n=0
        while s.step() is not None:n+=1
    with p.open() as f:rows=list(csv.DictReader(f))
    assert n==120
    assert len({float(r['t_gpst_s']) for r in rows if r['kind']=='R'})==120
