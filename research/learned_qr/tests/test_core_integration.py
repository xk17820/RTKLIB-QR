"""Test the actual top-level solver, not a header-only test stub."""
from pathlib import Path
import hashlib
import json
from tools.apply_patch import function_text
ROOT = Path(__file__).resolve().parents[3]
KIT = Path(__file__).resolve().parents[1]

def test_native_core_is_integrated():
    assert (ROOT/'src/learned_qr_adapter.h').exists(), 'Q/R is not integrated in native src/'
    assert '#include "learned_qr_adapter.h"' in (ROOT/'src/rtkpos.c').read_text()
    assert 'void *learned_qr;' in (ROOT/'src/rtklib.h').read_text()

def test_original_integer_search_and_dd_covariance_are_unchanged():
    expected = json.loads((KIT/'upstream_protected.json').read_text())
    text = (ROOT/'src/rtkpos.c').read_text()
    for name, digest in expected.items():
        data = (ROOT/'src/lambda.c').read_bytes() if name=='lambda.c' else function_text(text,name).encode()
        assert hashlib.sha256(data).hexdigest() == digest, name

def test_native_and_research_headers_match():
    for name in ('learned_qr.h','learned_qr_adapter.h','qr_bridge.c','qr_bridge.h','qr_trace.h','qr_filter.h'):
        assert (ROOT/'src'/name).exists(), 'native header is missing'
        assert (ROOT/'src'/name).read_bytes() == (KIT/'src'/name).read_bytes()
