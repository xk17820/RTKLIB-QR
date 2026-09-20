"""Actual native forwards for V1/V2/V3 at, above and below identity."""
import ctypes as C
import math
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from qrlearn.model import QRModel


@pytest.fixture(scope='module')
def evaluator(tmp_path_factory):
    path=tmp_path_factory.mktemp('export_boundary')
    source=path/'wrapper.c'
    source.write_text('''#include "learned_qr.h"
int evaluate(const char *path,const double *history,double *value) {
    qr_model model;
    if (!qr_load(&model,path,1)) return 0;
    return qr_model_forward(&model,0,history,value);
}
''')
    include=Path(__file__).resolve().parents[1]/'src'
    output=path/'wrapper.so'
    subprocess.run(['cc','-std=c99','-shared','-fPIC','-O2','-Wall','-Wextra',
                    '-Werror','-I'+str(include),str(source),'-lm','-o',str(output)],check=True)
    lib=C.CDLL(str(output));D=C.POINTER(C.c_double)
    lib.evaluate.argtypes=[C.c_char_p,D,D]
    lib.evaluate.restype=C.c_int
    return lib,D


@pytest.mark.parametrize('policy',['unconstrained','code_guard','scl','scl_blind'])
@pytest.mark.parametrize('head',[-.4,-1e-10,0.,1e-10,.4,20.])
def test_forward_boundary_and_gates_match_all_native_format_versions(tmp_path,evaluator,policy,head):
    lib,D=evaluator
    model=QRModel(r_policy=policy).double()
    with torch.no_grad():model.r.fc2.bias.fill_(head)
    path=tmp_path/'model.qr';model.export(path,provenance='synthetic')
    expected_magic={'unconstrained':'V1','code_guard':'V2','scl':'V3','scl_blind':'V3'}[policy]
    assert path.read_text().startswith('RTKLIB_QR_'+expected_magic+' 2 ')
    for cn0 in (0.,30.,45.):
        for evidence in (0.,2.,12.):
            for code in (0.,1.):
                for valid in (0.,1.):
                    h=np.zeros((10,13),dtype=np.float64)
                    h[:,0]=(cn0-40.)/10.;h[:,1]=.5
                    h[:,3]=code;h[:,8]=math.log1p(evidence);h[:,-1]=valid
                    out=np.zeros(1)
                    assert lib.evaluate(str(path).encode(),h.ctypes.data_as(D),out.ctypes.data_as(D))==1
                    expected=model.r(torch.tensor(h)).item()
                    assert out[0]==pytest.approx(expected,rel=3e-12,abs=3e-12)
                    if policy!='unconstrained':
                        assert 1. <= out[0] <= model.r.limit*(1.+1e-14)
                        if head<=0. or not code or not valid:assert out[0]==1.
