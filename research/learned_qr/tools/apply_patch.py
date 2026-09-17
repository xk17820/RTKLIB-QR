#!/usr/bin/env python3
"""Apply only to the inspected upstream rtkpos.c blob; fail closed on drift.

This script does not create a GitHub repository, authenticate or push anything.
It requires a clean local checkout on a non-main branch (or --create-branch).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

UPSTREAM_COMMIT='06e8644287ff07efc4c53bbf3e7f9dafb0355605'
RTKPOS_BLOB='edf17ff4e507687b597a0cfc2232db70099b8527'
UNCHANGED=('resamb_LAMBDA','manage_amb_LAMBDA','holdamb','ddcov')
ROOT=Path(__file__).resolve().parents[1]

def blob_sha(content: bytes) -> str:
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()

def function_range(text: str,name: str) -> tuple[int,int]:
    pattern=re.compile(r'(?m)^(?:static\s+(?:inline\s+)?|extern\s+)[\w\s*]+?\b'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{')
    matches=list(pattern.finditer(text))
    if len(matches)!=1: raise ValueError(f'expected exactly one function {name}, found {len(matches)}')
    match=matches[0];i=match.end()-1;depth=0;state='code'
    while i<len(text):
        ch=text[i];pair=text[i:i+2]
        if state=='block':
            if pair=='*/':state='code';i+=2;continue
        elif state=='line':
            if ch=='\n':state='code'
        elif state in ('"',"'"):
            if ch=='\\':i+=2;continue
            if ch==state:state='code'
        else:
            if pair=='/*':state='block';i+=2;continue
            if pair=='//':state='line';i+=2;continue
            if ch in ('"',"'"):state=ch
            elif ch=='{':depth+=1
            elif ch=='}':
                depth-=1
                if depth==0:return match.start(),i+1
        i+=1
    raise ValueError(f'unclosed function {name}')

def function_text(text: str,name: str) -> str:
    start,end=function_range(text,name);return text[start:end]

def once(text: str,old: str,new: str) -> str:
    if text.count(old)!=1:raise ValueError(f'anchor must occur once: {old!r}')
    return text.replace(old,new,1)

def transform_function(text: str,name: str,transform) -> str:
    a,b=function_range(text,name)
    return text[:a]+transform(text[a:b])+text[b:]

def patch_rtkpos(text: str,*,verify_sha: bool=True) -> str:
    if 'learned_qr_adapter.h' in text:raise ValueError('patch already present')
    if verify_sha and blob_sha(text.encode('utf-8'))!=RTKPOS_BLOB:
        raise ValueError(f'unrecognized upstream rtkpos.c; expected blob {RTKPOS_BLOB} at {UPSTREAM_COMMIT}')
    saved={name:function_text(text,name) for name in UNCHANGED}
    marker='/* global variables ----------------------------------------------------------*/'
    text=once(text,marker,'#include "learned_qr_adapter.h"\n'+marker)
    def variance(t):
        t=once(t,'varerr(int sat,','varerr(rtk_t *rtk, int sat,')
        t=once(t,'const obsd_t *obs)','const obsd_t *obs, const obsd_t *baseobs)')
        return once(t,'return var;',
            'return qr_apply_r(rtk,sat,sys,el,snr_rover,snr_base,bl,dt,f,opt,obs,baseobs,var);')
    text=transform_function(text,'varerr',variance)
    def dd(t):
        if len(re.findall(r'\bvarerr\(',t))!=4:raise ValueError('expected four SD variance calls in ddres')
        t=re.sub(r'\bvarerr\(', 'varerr(rtk,',t)
        for s,n in (('i',1),('j',3)):
            old=f'&obs[iu[{s}]])';new=f'&obs[iu[{s}]],&obs[ir[{s}]])'
            if t.count(old)!=n:raise ValueError(f'expected {n} variance observation arguments for {s}')
            t=t.replace(old,new)
        return t
    text=transform_function(text,'ddres',dd)
    def motion(t):
        t=once(t,'Q[8]=SQR(rtk->opt.prn[4])*fabs(tt);',
                 'Q[8]=SQR(rtk->opt.prn[4])*fabs(tt);\n    qr_apply_q(rtk,tt,Q);')
        for marker in ('if (norm(rtk->x, 3) <= RE_WGS84 / 2) {','if (var>VAR_POS) {'):
            t=once(t,marker,marker+'\n        qr_rtk_reset(rtk->learned_qr);')
        return t
    text=transform_function(text,'udpos',motion)
    text=transform_function(text,'rtkinit',lambda t:once(t,'rtk->intpres_nb=0;',
        'rtk->intpres_nb=0;\n    rtk->learned_qr=qr_rtk_create();'))
    text=transform_function(text,'rtkfree',lambda t:once(t,'rtk->nx=rtk->na=0;',
        'qr_rtk_destroy(rtk->learned_qr);\n    rtk->learned_qr=NULL;\n    rtk->nx=rtk->na=0;'))
    for name,source in saved.items():
        if function_text(text,name)!=source:raise ValueError(f'protected function changed: {name}')
    return text

def patch_header(text: str) -> str:
    if 'learned_qr' in text:raise ValueError('rtk_t already contains a learned_qr extension')
    matches=list(re.finditer(r'}\s*rtk_t\s*;',text))
    if len(matches)!=1:raise ValueError('expected one rtk_t typedef')
    a=matches[0].start()
    return text[:a]+'    void *learned_qr; /* per-filter QR context; rebuild ALL dependent binaries */\n'+text[a:]

def git(repo: Path,*args: str) -> str:
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--check',action='store_true',help='validate only; do not write')
    parser.add_argument('--create-branch',action='store_true')
    args=parser.parse_args();repo=args.repo.resolve()
    if repo==ROOT or repo in ROOT.parents:
        raise SystemExit('Keep the kit OUTSIDE the destination checkout during installation.')
    cpath=repo/'src/rtkpos.c';hpath=repo/'src/rtklib.h'
    cbytes=cpath.read_bytes()
    if blob_sha(cbytes)!=RTKPOS_BLOB:
        raise SystemExit(f'Wrong upstream source. Check out commit {UPSTREAM_COMMIT}; refusing to guess.')
    ctext=cbytes.decode('utf-8');htext=hpath.read_bytes().decode('utf-8')
    modified_c=patch_rtkpos(ctext);modified_h=patch_header(htext)
    report={'upstream_commit':UPSTREAM_COMMIT,'rtkpos_source_blob':RTKPOS_BLOB,
            'protected_function_sha256':{n:hashlib.sha256(function_text(ctext,n).encode()).hexdigest() for n in UNCHANGED},
            'modified_files':['src/rtkpos.c','src/rtklib.h'],
            'new_headers':['src/learned_qr.h','src/learned_qr_adapter.h'],
            'full_rtklib_build_verified':False,'rinex_regression_verified':False}
    if args.check:
        print(json.dumps(report,indent=2));return
    if git(repo,'status','--porcelain'):
        raise SystemExit('Destination has uncommitted/untracked work. Use a clean independent checkout.')
    branch=git(repo,'branch','--show-current')
    if args.create_branch:
        git(repo,'switch','-c','research/learned-qr')
    elif not branch or branch in ('main','master'):
        raise SystemExit('Use --create-branch or an existing independent branch; main/master are not modified.')
    target=repo/'research/learned_qr'
    if target.exists():raise SystemExit('research/learned_qr already exists; refusing to overwrite')
    for name in ('learned_qr.h','learned_qr_adapter.h'):
        if (repo/'src'/name).exists():raise SystemExit(f'src/{name} exists; refusing to overwrite')
    # All source anchors and checks have succeeded before writing the first file.
    shutil.copytree(ROOT,target,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','build','outputs','*.zip'))
    for name in ('learned_qr.h','learned_qr_adapter.h'):
        shutil.copy2(ROOT/'src'/name,repo/'src'/name)
    cpath.write_bytes(modified_c.encode('utf-8'));hpath.write_bytes(modified_h.encode('utf-8'))
    (target/'PATCH_APPLIED.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Applied locally. No commit or push performed. Rebuild all RTKLIB binaries and run independent RINEX tests.')

if __name__=='__main__':main()
