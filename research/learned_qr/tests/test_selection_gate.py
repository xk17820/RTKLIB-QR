import copy
import importlib.util
import pytest


def get():
    assert importlib.util.find_spec('qrlearn.selection'), 'native selection missing'
    from qrlearn.selection import deployment_decision
    return deployment_decision


def baseline():
    return dict(aggregate={'rms_3d_m':2.},routes={'a':dict(attempted_epochs=10,matched_epochs=10,
       solution_available_epochs=10,matched_times=list(range(10)),available_times=list(range(10)),
       rms_3d_m=2.,p95_3d_m=3.,solution_fixed_fraction=.8,fixed_large_error_proxy_count=0)})


def test_identity_and_worse_never_win():
    f=get();b=baseline();c=copy.deepcopy(b);c['aggregate']['rms_3d_m']=3
    assert not f(b,b)['accepted'];assert not f(b,c)['accepted']


@pytest.mark.parametrize('key,value',[('rms_3d_m',2.5),('p95_3d_m',3.5),
 ('solution_fixed_fraction',.6),('fixed_large_error_proxy_count',1),
 ('matched_times',list(range(1,11))),('solution_available_epochs',9)])
def test_aggregate_gain_cannot_hide_regression(key,value):
    f=get();b=baseline();c=copy.deepcopy(b);c['aggregate']['rms_3d_m']=1.5
    c['routes']['a'][key]=value
    assert not f(b,c)['accepted']


def test_good_and_empty_validation():
    f=get();b=baseline();c=copy.deepcopy(b);c['aggregate']['rms_3d_m']=1.5
    c['routes']['a'].update(rms_3d_m=1.5,p95_3d_m=2.5)
    assert f(b,c)['accepted']
    assert not f(b,dict(aggregate={'rms_3d_m':None},routes={}))['accepted']
    c['aggregate']['rms_3d_m']=float('nan');assert not f(b,c)['accepted']
