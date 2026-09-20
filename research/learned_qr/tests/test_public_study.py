"""Metadata and reference conversion are independent of solver outputs."""
from pathlib import Path
import json
import numpy as np
import pytest
from qrlearn import public_study as p


def test_reference_rotation_uses_publisher_forward_right_up_convention():
    np.testing.assert_allclose(p.span_body_to_enu(0,0,0),np.eye(3),atol=1e-15)
    np.testing.assert_allclose(p.span_body_to_enu(0,0,90)@np.array([0.,1.,0.]),[1.,0.,0.],atol=1e-15)
    r=p.span_body_to_enu(6,-4,121)
    np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-14)
    assert np.linalg.det(r)==pytest.approx(1.)


def test_gnss_to_reference_arm_is_subtracted_not_added():
    x=p.llh_to_ecef(22.,114.,40.)
    y=p.reference_to_splitter_antenna(x,0.,0.,0.)
    np.testing.assert_allclose(p.enu_rotation(x)@(y-x),[0.,.560,.070],atol=1e-9)
    z=p.reference_to_splitter_antenna(x,0.,0.,90.)
    np.testing.assert_allclose(p.enu_rotation(x)@(z-x),[.560,0.,.070],atol=1e-9)


def test_dms_negative_sign_applies_to_entire_angle():
    assert p.dms_to_degrees(-22,30,0)==-22.5
    assert p.dms_to_degrees(22,30,0)==22.5
    with pytest.raises(ValueError):p.dms_to_degrees(22,60,0)


def test_hk_reference_preserves_gpst_no_utc_shift(tmp_path):
    src=tmp_path/'raw.txt';src.write_text('header\nheader\n1621218775 2158 95593 22 0 0 114 0 0 40 0 0 0 0 0 0 0 0 0 2\n')
    dst=tmp_path/'converted.csv';m=p.convert_hk_reference(src,dst)
    r=p.Reference.load(dst)
    assert r.time[0]==2158*604800+95593
    assert m['source_quality_counts']=={'2':1}
    assert m['lever_applied_to_reference_only'] is True
    np.testing.assert_allclose(r.xyz[0],p.reference_to_splitter_antenna(p.llh_to_ecef(22,114,40),0,0,0),atol=1e-8)


def test_whole_campaign_and_duplicate_bytes_cannot_cross_partitions(tmp_path):
    a=tmp_path/'a.obs';b=tmp_path/'b.obs';a.write_text('same');b.write_text('same')
    routes={'train':[{'id':'a','campaign':'day1','rover':str(a)}],'validation':[{'id':'b','campaign':'day2','rover':str(b)}]}
    with pytest.raises(ValueError,match='content'):p.validate_partitions(routes)
    b.write_text('different');routes['validation'][0]['campaign']='day1'
    with pytest.raises(ValueError,match='campaign'):p.validate_partitions(routes)
    routes['validation'][0]['campaign']='day2'
    assert p.validate_partitions(routes)['route_count']==2


def test_regular_slots_keep_missing_observations_and_do_not_retime_measurements():
    r=p.scheduled_slots([100.006,101.005,104.004],1.)
    assert r['scheduled_slots']==5 and r['missing_observation_slots']==2
    assert r['row_slots']==[0,1,4]
    assert r['original_times']==[100.006,101.005,104.004]
    with pytest.raises(ValueError,match='duplicate'):p.scheduled_slots([100.,100.001],1.)


def test_rinex_inventory_reads_complete_time_span(tmp_path):
    src=tmp_path/'a.obs';src.write_text('     3.03           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n                                                            END OF HEADER\n> 2021 05 17 02 33 13.0060000  0  0\n> 2021 05 17 02 33 16.0060000  0  0\n')
    r=p.rinex_inventory(src,1.)
    assert r['raw_epochs']==2 and r['scheduled_slots']==4
    assert r['first_gpst_s']==2158*604800+95593.006


def test_rinex_header_event_without_timestamp_is_not_an_observation(tmp_path):
    src=tmp_path/'events.obs'
    src.write_text('                                                            END OF HEADER\n> 2021 05 17 02 33 13.0060000  0  0\n>                              4  1\nnew header                                                  COMMENT\n> 2021 05 17 02 33 16.0060000  0  0\n')
    r=p.rinex_inventory(src,1.)
    assert r['raw_epochs']==2 and r['scheduled_slots']==4
