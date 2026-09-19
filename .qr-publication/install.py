"""One-time, checksum-pinned publication of the locally tested research patch."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

DIGEST = '6fab858d335a8294eab7786ed0e8c4979dd1b1f8b45155c5f2bfd89e75e24bf3'
REPO = 'xk17820/RTKLIB-QR'
BRANCH = 'research/learned-qr'


def git(root, *args, data=None):
    return subprocess.check_output(['git', '-C', str(root), *args], input=data)


def check_path(path):
    p = PurePosixPath(path)
    if (not path or p.is_absolute() or '..' in p.parts or '\\' in path or
            '\x00' in path or str(p) != path or
            not (path == 'README-QR.md' or path.startswith('src/') or
                 path.startswith('research/learned_qr/') or
                 path.startswith('docs/superpowers/'))):
        raise RuntimeError('unapproved patch path: ' + repr(path))


def check_hashes(root, manifest, key):
    for entry in manifest:
        check_path(entry['path'])
        p = root / entry['path']
        for q in [p, *p.parents]:
            if q == root:
                break
            if q.is_symlink():
                raise RuntimeError('symlink in patch target')
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if actual != entry[key] or (actual is None and p.exists()):
            raise RuntimeError(key + ' hash mismatch: ' + entry['path'])


def manifest_path(root):
    return root / git(root, 'rev-parse', '--git-path', 'qr-publication-manifest.json').decode().strip()


def apply(root):
    root = Path(root).resolve()
    data = b''.join((root / '.qr-publication' / f'part{i}').read_bytes() for i in range(4))
    if hashlib.sha256(data).hexdigest() != DIGEST:
        raise RuntimeError('publication checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:xz') as tar:
        members = tar.getmembers()
        if {m.name for m in members} != {'manifest.json', 'implementation.patch'} or len(members) != 2:
            raise RuntimeError('unexpected archive members')
        if any(not m.isfile() or m.size > 1000000 for m in members):
            raise RuntimeError('unexpected archive entry type/size')
        manifest = json.loads(tar.extractfile('manifest.json').read())
        patch = tar.extractfile('implementation.patch').read()
    if len(manifest) != 40 or len({e['path'] for e in manifest}) != 40:
        raise RuntimeError('unexpected manifest count')
    check_hashes(root, manifest, 'before')
    git(root, 'diff', '--quiet')
    git(root, 'diff', '--cached', '--quiet')
    stats = git(root, 'apply', '--numstat', '-z', '-', data=patch)
    paths = [r.decode().split('\t', 2)[2] for r in stats.split(b'\0') if r]
    if sorted(paths) != sorted(e['path'] for e in manifest):
        raise RuntimeError('patch/manifest path mismatch')
    git(root, 'apply', '--check', '-', data=patch)
    git(root, 'apply', '-', data=patch)
    check_hashes(root, manifest, 'after')
    manifest_path(root).write_text(json.dumps(manifest), encoding='utf-8')
    print('APPLIED AND VERIFIED: 40 exact source files')
    return manifest


def publish(root):
    root = Path(root).resolve()
    if os.environ.get('GITHUB_REPOSITORY') != REPO or os.environ.get('GITHUB_REF') != 'refs/heads/' + BRANCH:
        raise RuntimeError('publication restricted to the approved research branch')
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    if head != os.environ.get('GITHUB_SHA'):
        raise RuntimeError('checkout moved during validation')
    remote = git(root, 'ls-remote', 'origin', 'refs/heads/' + BRANCH).decode().split()[0]
    if remote != head:
        raise RuntimeError('remote moved; refusing to overwrite concurrent work')
    manifest = json.loads(manifest_path(root).read_text())
    check_hashes(root, manifest, 'after')
    git(root, 'add', '-f', '--', *(e['path'] for e in manifest))
    git(root, 'rm', '--', *(f'.qr-publication/part{i}' for i in range(4)), '.qr-publication/install.py')
    staged = git(root, 'diff', '--cached', '--name-only', '-z').decode().split('\0')
    expected = {e['path'] for e in manifest} | {f'.qr-publication/part{i}' for i in range(4)} | {'.qr-publication/install.py'}
    if set(filter(None, staged)) != expected:
        raise RuntimeError('unexpected staged publication paths')
    git(root, 'config', 'user.name', 'github-actions[bot]')
    git(root, 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    git(root, 'commit', '-m', 'feat: complete live reference-loss Q/R training and native C replay')
    git(root, 'push', 'origin', 'HEAD:refs/heads/' + BRANCH)
    sha = git(root, 'rev-parse', 'HEAD').decode().strip()
    remote = git(root, 'ls-remote', 'origin', 'refs/heads/' + BRANCH).decode().split()[0]
    if remote != sha:
        raise RuntimeError('remote SHA verification failed')
    print('PUSH VERIFIED:', sha)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) != 2 or sys.argv[1] not in {'apply', 'publish'}:
        raise SystemExit('usage: install.py apply|publish')
    (apply if sys.argv[1] == 'apply' else publish)(root)
