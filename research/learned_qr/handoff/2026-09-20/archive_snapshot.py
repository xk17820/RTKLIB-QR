"""Save the exact unfinished SCL source and its failed/passed checks; not a release."""
from pathlib import Path, PurePosixPath
import hashlib, json, lzma, os, re, subprocess, sys

REPO = 'xk17820/RTKLIB-QR'
BRANCH = 'research/learned-qr'
MAIN = '06e8644287ff07efc4c53bbf3e7f9dafb0355605'
DEST = 'research/learned_qr/handoff/2026-09-20'
PAYLOAD = 'f96785b2bd811259a911ffc9b6da3e540233937dbd9de3734983634ebdda8434'
PATCH = '63f31bf95c53d9701db56bc06cf6c990fb506e0678a2330651b6398f43fdb61c'

def git(root, *args, data=None, env=None):
    return subprocess.check_output(['git', '-C', str(root), *args], input=data, env=env)

def checked_path(root, name):
    p = PurePosixPath(name)
    if (not isinstance(name, str) or not name or p.is_absolute() or '..' in p.parts or
        '\\' in name or '\x00' in name or str(p) != name or
        not (name == 'README-QR.md' or name.startswith(('src/', 'research/learned_qr/')))):
        raise RuntimeError('unapproved source path: ' + repr(name))
    target = root / name
    for ancestor in [target, *target.parents]:
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise RuntimeError('symlink source rejected')
    return target

def verify(root, entries):
    for name, expected in entries.items():
        p = checked_path(root, name)
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if actual != expected or (expected is None and p.exists()):
            raise RuntimeError('source hash mismatch: ' + name)

def apply(root):
    here = root / '.scl-publication'
    manifest = json.loads((here / 'manifest.json').read_text())
    if len(manifest['post']) != 27 or set(manifest['pre']) != set(manifest['post']):
        raise RuntimeError('unexpected manifest')
    data = b''.join((here / f'part{i}').read_bytes() for i in range(4))
    if hashlib.sha256(data).hexdigest() != PAYLOAD or manifest['payload_sha256'] != PAYLOAD:
        raise RuntimeError('payload checksum mismatch')
    decoder = lzma.LZMADecompressor()
    patch = decoder.decompress(data, max_length=3000001)
    if not decoder.eof or len(patch) > 3000000 or hashlib.sha256(patch).hexdigest() != PATCH or manifest['patch_sha256'] != PATCH:
        raise RuntimeError('patch checksum/size mismatch')
    verify(root, manifest['pre'])
    git(root, 'diff', '--quiet')
    git(root, 'diff', '--cached', '--quiet')
    entries = git(root, 'apply', '--numstat', '-z', '-', data=patch).split(b'\0')
    names = [e.decode().split('\t', 2)[2] for e in entries if e]
    if sorted(names) != sorted(manifest['post']):
        raise RuntimeError('patch/manifest mismatch')
    git(root, 'apply', '--check', '-', data=patch)
    git(root, 'apply', '-', data=patch)
    verify(root, manifest['post'])
    evidence = root / DEST / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (evidence / 'scl_source.patch').write_bytes(patch)
    (evidence / 'scl_payload.xz').write_bytes(data)
    old = root / '.qr-repair'
    if old.is_dir():
        parts = sorted(old.glob('part*'), key=lambda p: int(p.name[4:]))
        (evidence / 'previous_guard_payload.bin').write_bytes(b''.join(p.read_bytes() for p in parts))
        if (old / 'manifest.json').is_file():
            (evidence / 'previous_guard_manifest.json').write_bytes((old / 'manifest.json').read_bytes())
    print('WIP SOURCE MATERIALIZED: 27 files, exact post-image hashes verified; this is not a test-pass claim.')

def record(root):
    evidence = root / DEST / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    paths = {}
    for p in sorted(evidence.iterdir()):
        if p.is_file():
            paths[p.name] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    status = {'purpose': 'archive unfinished work for a new chat, not approve deployment',
              'source_commit_before_snapshot': os.environ.get('GITHUB_SHA'),
              'run_url': 'https://github.com/' + REPO + '/actions/runs/' + os.environ.get('GITHUB_RUN_ID', 'unknown'),
              'source_materialized': (evidence / 'source_manifest.json').exists(),
              'checks': {k: os.environ.get(k.upper() + '_OUTCOME', 'not_run') for k in ('dependencies', 'build', 'ctest', 'pytest')},
              'files': paths, 'new_large_experiments_completed': False, 'production_approved': False}
    (root / DEST / 'SNAPSHOT_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status['checks']))

def publish(root):
    if os.environ.get('GITHUB_REPOSITORY') != REPO or os.environ.get('GITHUB_REF') != 'refs/heads/' + BRANCH:
        raise RuntimeError('snapshot destination is not approved')
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    if head != os.environ.get('GITHUB_SHA'):
        raise RuntimeError('checkout changed')
    evidence = root / DEST / 'evidence'
    manifest = json.loads((evidence / 'source_manifest.json').read_text())
    verify(root, manifest['post'])
    token = os.environ.get('SNAPSHOT_TOKEN')
    if not token:
        raise RuntimeError('missing scoped publication token')
    import base64
    auth = base64.b64encode(('x-access-token:' + token).encode()).decode()
    env = {**os.environ, 'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader', 'GIT_CONFIG_VALUE_0': 'AUTHORIZATION: basic ' + auth}
    def ref(branch):
        return git(root, 'ls-remote', 'origin', 'refs/heads/' + branch, env=env).decode().split()[0]
    if ref(BRANCH) != head or ref('main') != MAIN:
        raise RuntimeError('remote changed; refusing overwrite')
    git(root, 'add', '-f', '--', *manifest['post'], DEST + '/evidence', DEST + '/SNAPSHOT_STATUS.json')
    for directory in ('.scl-publication', '.qr-repair'):
        if (root / directory).exists():
            git(root, 'rm', '-r', '--', directory)
    staged = git(root, 'diff', '--cached', '--name-only', '-z').decode().split('\0')
    for name in filter(None, staged):
        if name not in manifest['post'] and not name.startswith((DEST + '/', '.scl-publication/', '.qr-repair/')):
            raise RuntimeError('unexpected staged path: ' + name)
    git(root, 'config', 'user.name', 'github-actions[bot]')
    git(root, 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    git(root, 'commit', '-m', 'chore: archive unfinished SCL source, test evidence and research handoff (not a release)')
    git(root, 'push', 'origin', 'HEAD:refs/heads/' + BRANCH, env=env)
    final = git(root, 'rev-parse', 'HEAD').decode().strip()
    if ref(BRANCH) != final or ref('main') != MAIN:
        raise RuntimeError('published refs not verified')
    print('SNAPSHOT PUSH VERIFIED: ' + final + '; main unchanged; see SNAPSHOT_STATUS.json for failures.')

if __name__ == '__main__':
    root = Path.cwd().resolve()
    if len(sys.argv) != 2 or sys.argv[1] not in ('apply', 'record', 'publish'):
        raise SystemExit('usage: archive_snapshot.py apply|record|publish')
    globals()[sys.argv[1]](root)
