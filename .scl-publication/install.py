"""One-time, checksum-pinned source publication on the user-approved branch only."""
from pathlib import Path
import base64,hashlib,json,lzma,os,subprocess,sys
ROOT=Path.cwd();HERE=ROOT/'.scl-publication';REPO='xk17820/RTKLIB-QR';BRANCH='research/learned-qr'
def run(*args,**kwargs):return subprocess.check_output(args,text=True,**kwargs).strip()
def verify(items):
 for name,wanted in items.items():
  p=Path(name)
  if p.is_absolute() or '..'in p.parts or not(name.startswith(('src/','research/learned_qr/')) or name=='README-QR.md'):raise RuntimeError('unapproved source path')
  p=ROOT/p
  if p.is_symlink():raise RuntimeError('symlink source rejected')
  actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
  if actual!=wanted:raise RuntimeError('source checksum differs: '+name)
def main(action):
 m=json.loads((HERE/'manifest.json').read_text())
 if action=='apply':
  verify(m['pre']);data=b''.join((HERE/f'part{i}').read_bytes() for i in range(4))
  if hashlib.sha256(data).hexdigest()!=m['payload_sha256']:raise RuntimeError('payload checksum')
  patch=lzma.decompress(data)
  if len(patch)>3000000 or hashlib.sha256(patch).hexdigest()!=m['patch_sha256']:raise RuntimeError('patch checksum/size')
  subprocess.run(['git','apply','--check','-'],input=patch,check=True)
  subprocess.run(['git','apply','-'],input=patch,check=True);verify(m['post'])
  print('EXACT SOURCE VERIFIED',len(m['post']))
 elif action=='publish':
  if os.environ.get('GITHUB_REPOSITORY')!=REPO or os.environ.get('GITHUB_REF')!='refs/heads/'+BRANCH:raise RuntimeError('unapproved destination')
  verify(m['post']);head=run('git','rev-parse','HEAD')
  if head!=os.environ.get('GITHUB_SHA'):raise RuntimeError('checkout changed')
  token=os.environ.get('PUBLISH_TOKEN')
  if not token:raise RuntimeError('missing scoped token')
  auth=base64.b64encode(('x-access-token:'+token).encode()).decode()
  env={**os.environ,'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'http.https://github.com/.extraheader','GIT_CONFIG_VALUE_0':'AUTHORIZATION: basic '+auth}
  if run('git','ls-remote','origin','refs/heads/'+BRANCH,env=env).split()[0]!=head:raise RuntimeError('remote advanced; no force')
  subprocess.run(['git','add','--',*m['post']],check=True)
  subprocess.run(['git','rm','-r','--','.scl-publication'],check=True)
  if (ROOT/'.qr-repair').exists():subprocess.run(['git','rm','-r','--','.qr-repair'],check=True)
  names=run('git','diff','--cached','--name-only').splitlines()
  if any(n not in m['post'] and not n.startswith(('.scl-publication/','.qr-repair/')) for n in names):raise RuntimeError('unexpected staged file')
  run('git','config','user.name','github-actions[bot]');run('git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com')
  run('git','commit','-m','feat: SCL-RTK conditional-candidate training, causal covariance features and DGPS supervision')
  subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],check=True,env=env)
  final=run('git','rev-parse','HEAD')
  if run('git','ls-remote','origin','refs/heads/'+BRANCH,env=env).split()[0]!=final:raise RuntimeError('published ref mismatch')
  print('PUSH VERIFIED',final)
 else:raise ValueError('apply or publish required')
if __name__=='__main__':main(sys.argv[1])
