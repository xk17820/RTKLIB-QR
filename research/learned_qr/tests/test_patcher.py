from pathlib import Path
import importlib.util
import pytest

# Small interface fixtures based on the inspected upstream functions, NOT a copy
# of all RTKLIB and NOT evidence of a full upstream compile or RINEX regression.
C='''#include "rtklib.h"
#define NF(opt) ((opt)->nf)
/* global variables ----------------------------------------------------------*/
static double varerr(int sat, int sys, double el, double snr_rover, double snr_base,
                     double bl, double dt, int f, const prcopt_t *opt, const obsd_t *obs)
{
    double var=1;
    return var;
}
static void udpos(rtk_t *rtk, double tt)
{
    if (norm(rtk->x, 3) <= RE_WGS84 / 2) {
        initx(rtk,1,1,0);
    }
    if (var>VAR_POS) {
        initx(rtk,1,1,0); return;
    }
    Q[8]=SQR(rtk->opt.prn[4])*fabs(tt);
}
static int ddres(rtk_t *rtk)
{
    refvar=varerr(sat[j],sysj,el,snr1,snr2,bl,dt,f,opt,&obs[iu[j]]);
    refvar=varerr(sat[j],sysj,el,snr1,snr2,bl,dt,f,opt,&obs[iu[j]]);
    Ri[nv]=varerr(sat[i],sysi,el,snr1,snr2,bl,dt,f,opt,&obs[iu[i]]);
    Rj[nv]=varerr(sat[j],sysj,el,snr1,snr2,bl,dt,f,opt,&obs[iu[j]]);
    return 0;
}
extern void rtkinit(rtk_t *rtk, const prcopt_t *opt)
{
    rtk->intpres_nb=0;
}
extern void rtkfree(rtk_t *rtk)
{
    rtk->nx=rtk->na=0;
    free(rtk->x);
}
static int resamb_LAMBDA(rtk_t *rtk) { char *s="}"; /* { */ return lambda(1); }
static int manage_amb_LAMBDA(rtk_t *rtk) { return resamb_LAMBDA(rtk); }
static void holdamb(rtk_t *rtk) { hold(rtk); }
static void ddcov(int n) { unchanged(n); }
'''

def module():
    assert importlib.util.find_spec('tools.apply_patch') is not None,'patcher must be implemented'
    from tools import apply_patch
    return apply_patch

def test_correct_hook_locations_and_four_variance_calls():
    m=module();p=m.patch_rtkpos(C,verify_sha=False)
    assert '#include "learned_qr_adapter.h"' in p
    assert p.count('varerr(rtk,')==4
    assert 'static double varerr(rtk_t *rtk,' in p
    assert p.count('&obs[ir[j]]')==3 and p.count('&obs[ir[i]]')==1
    assert p.count('qr_apply_q(rtk,tt,Q)')==1
    assert p.count('qr_rtk_reset(rtk->learned_qr)')==2
    assert 'rtk->learned_qr=qr_rtk_create()' in p
    assert 'qr_rtk_destroy(rtk->learned_qr)' in p

def test_integer_functions_and_ddcov_remain_byte_identical():
    m=module();p=m.patch_rtkpos(C,verify_sha=False)
    for name in m.UNCHANGED:
        assert m.function_text(p,name)==m.function_text(C,name)

def test_unknown_source_and_missing_anchor_rejected():
    m=module()
    with pytest.raises(ValueError,match='upstream'):m.patch_rtkpos(C)
    with pytest.raises(ValueError):m.patch_rtkpos(C.replace('rtk->intpres_nb=0;','different();'),verify_sha=False)

def test_header_extension_and_double_application_rejected():
    m=module();h='typedef struct { int nx; } rtk_t;\n';p=m.patch_header(h)
    assert 'void *learned_qr;' in p
    with pytest.raises(ValueError):m.patch_header(p)
    with pytest.raises(ValueError):m.patch_rtkpos(m.patch_rtkpos(C,verify_sha=False),verify_sha=False)
