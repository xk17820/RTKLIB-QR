"""Bundled real RINEX inputs for software tests; not a reference trajectory."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def raw_kwargs():
    d=ROOT/'test/data/rinex'
    return dict(config=ROOT/'research/learned_qr/configs/short-baseline.conf',
        rover=d/'30400920.05o',base=d/'07590920.05o',
        nav=[d/'30400920.05n',d/'07590920.05n'],max_gap=31.,pair_age=.05)
