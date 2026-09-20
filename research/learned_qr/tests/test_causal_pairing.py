"""Observation availability, not proximity, determines base pairing.

Replacing predecessor selection with nearest-neighbour pairing must fail these
regressions. Observations and navigation are the unmodified bundled fixture;
only complete epoch blocks are selected, not reference-derived measurements.
"""
import math
from pathlib import Path
import re
import numpy as np
import pytest
from qrlearn.native import Session
from tests_support import raw_kwargs


def subset(source, target, indices):
    lines=Path(source).read_text().splitlines(keepends=True)
    pattern=re.compile(r'^ \d{2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\.\d+\s+[0-6]\s*\d+')
    starts=[i for i,line in enumerate(lines) if pattern.match(line)]
    assert len(starts)>max(indices)
    bounds=starts+[len(lines)]
    target.write_text(''.join(lines[:starts[0]])+''.join(''.join(lines[bounds[i]:bounds[i+1]]) for i in indices))
    return target


def setup(tmp_path, rover_indices, base_indices):
    kw=raw_kwargs();conf=tmp_path/'causal.conf'
    conf.write_text(kw['config'].read_text()+'\npos2-maxage=31\n')
    return {**kw,'config':conf,'rover':subset(kw['rover'],tmp_path/'rover.obs',rover_indices),
            'base':subset(kw['base'],tmp_path/'base.obs',base_indices),'pair_age':31.}


def test_future_base_is_unavailable_even_within_age_limit(tmp_path):
    kw=setup(tmp_path,[0],[1,2])
    with Session(**kw,training=False) as session:
        row=session.step()
        assert row['status']==0, 'a future observation is not an available RTK correction'
        assert np.isnan(row['position']).all()
        assert math.isnan(row['age']), 'there is no selected base epoch yet'
        assert session.step() is None


@pytest.mark.parametrize('future_index',[2,3,4])
def test_future_base_suffix_cannot_change_prefix_solution(tmp_path,future_index):
    kw=setup(tmp_path,[1],[0,future_index]);only_past=tmp_path/'past.obs'
    subset(raw_kwargs()['base'],only_past,[0])
    with Session(**kw,training=False) as session:full=session.step()
    with Session(**{**kw,'base':only_past},training=False) as session:prefix=session.step()
    assert full['age']==pytest.approx(30.)
    assert full['age']==prefix['age']
    assert full['status']==prefix['status']
    np.testing.assert_array_equal(full['position'],prefix['position'])


def test_equal_timestamp_is_available_and_restart_reproducible(tmp_path):
    kw=setup(tmp_path,[0,1],[0,1,2])
    with Session(**kw,training=False) as session:
        a=[session.step(),session.step()];session.restart();b=[session.step(),session.step()]
    for left,right in zip(a,b):
        assert left['age']==right['age']==0.
        assert left['status']==right['status']
        np.testing.assert_array_equal(left['position'],right['position'])


def test_explicit_zero_latency_retains_strict_past_only_fixture_replay():
    from tests_support import raw_kwargs
    from qrlearn.native import Session
    with Session(**raw_kwargs(),mode='off',training=False,pairing_latency=0.) as s:
        rows=[]
        while (r:=s.step()) is not None:rows.append(r)
    assert len(rows)==120
    assert all(r['age']>=0. or np.isnan(r['age']) for r in rows)
    assert all(r['release_t_gpst_s']>=r['time'] for r in rows)


def test_default_alignment_horizon_is_reported_not_hidden_lookahead():
    from tests_support import raw_kwargs
    from qrlearn.native import Session
    with Session(**raw_kwargs(),mode='off',training=False) as s:
        rows=[]
        while (r:=s.step()) is not None:rows.append(r)
    assert len(rows)==120 and all(r['status']>0 for r in rows)
    assert min(r['age'] for r in rows)<0.
    for r in rows:
        assert r['release_t_gpst_s']==pytest.approx(r['time']+max(0.,-r['age']),abs=1e-6)
        assert r['timestamp_horizon_s']<=.025000001
