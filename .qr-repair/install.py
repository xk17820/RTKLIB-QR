"""One-time exact-source transport; only the approved research branch is writable."""
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path.cwd()
HERE=ROOT/'.qr-repair'
REPO='xk17820/RTKLIB-QR'
BRANCH='research/learned-qr'

def command(*args,**kwargs):
    return subprocess.check_output(args,text=True,**kwargs).strip()

def verify(expected):
    for name,wanted in expected.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not (name.startswith(('research/learned_qr/','src/learned_qr','docs/superpowers/')) or name=='README-QR.md'):
            raise RuntimeError('invalid source path')
        p=ROOT/p
        if p.is_symlink():raise RuntimeError('symlink source rejected')
        actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        if actual!=wanted:raise RuntimeError('source hash mismatch: '+name)

def main(action):
    m=json.loads((HERE/'manifest.json').read_text())
    if action=='apply':
        verify(m['pre'])
        data=b''.join((HERE/f'part{i}').read_bytes() for i in range(8))
        if hashlib.sha256(data).hexdigest()!=m['payload_sha256']:raise RuntimeError('payload hash mismatch')
        patch=lzma.decompress(data)
        if len(patch)>1000000:raise RuntimeError('unexpected patch size')
        subprocess.run(['git','apply','--check','-'],input=patch,check=True)
        subprocess.run(['git','apply','-'],input=patch,check=True)
        verify(m['post'])
        print('APPLIED AND VERIFIED',len(m['post']),'exact source files')
    elif action=='publish':
        if os.environ.get('GITHUB_REPOSITORY')!=REPO or os.environ.get('GITHUB_REF')!='refs/heads/'+BRANCH:
            raise RuntimeError('wrong repository or branch')
        verify(m['post'])
        head=command('git','rev-parse','HEAD')
        if head!=os.environ.get('GITHUB_SHA'):raise RuntimeError('checkout changed')
        token=os.environ.get('PUBLISH_TOKEN')
        if not token:raise RuntimeError('missing scoped publication token')
        authorization=base64.b64encode(('x-access-token:'+token).encode()).decode()
        env={**os.environ,'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'http.https://github.com/.extraheader',
             'GIT_CONFIG_VALUE_0':'AUTHORIZATION: basic '+authorization}
        remote=command('git','ls-remote','origin','refs/heads/'+BRANCH,env=env).split()[0]
        if remote!=head:raise RuntimeError('branch moved; never force push')
        subprocess.run(['git','add','--',*m['post']],check=True)
        subprocess.run(['git','rm','-r','--','.qr-repair'],check=True)
        staged=command('git','diff','--cached','--name-only').splitlines()
        if any(p not in m['post'] and not p.startswith('.qr-repair/') for p in staged):
            raise RuntimeError('unexpected staged source')
        command('git','config','user.name','github-actions[bot]')
        command('git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com')
        command('git','commit','-m','fix: anchor phase/code R, validate native RTK, and record frozen holdout results')
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],check=True,env=env)
        final=command('git','rev-parse','HEAD')
        remote=command('git','ls-remote','origin','refs/heads/'+BRANCH,env=env).split()[0]
        if final!=remote:raise RuntimeError('published ref mismatch')
        print('PUSH VERIFIED',final)
    else:raise ValueError('action must be apply or publish')

if __name__=='__main__':main(sys.argv[1])
