"""Publication safety tests, not scientific or RTK performance tests."""
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import subprocess
import pytest

spec = importlib.util.spec_from_file_location('archive_snapshot', Path(__file__).with_name('archive_snapshot.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def run(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])

@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'repo'; root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    run(root, 'config', 'user.name', 'Archive test')
    run(root, 'config', 'user.email', 'test@local.invalid')
    pre = {}; post = {}
    for i in range(27):
        name = f'src/test_{i}.txt'
        p = root / name; p.parent.mkdir(exist_ok=True)
        p.write_text(f'old {i}\n')
        pre[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    run(root, 'add', '.'); run(root, 'commit', '-qm', 'base')
    for i in range(27):
        p = root / f'src/test_{i}.txt'; p.write_text(f'new {i}\n')
        post[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    patch = run(root, 'diff', '--binary')
    run(root, 'checkout', '--', 'src')
    payload = lzma.compress(patch)
    monkeypatch.setattr(m, 'PAYLOAD', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(m, 'PATCH', hashlib.sha256(patch).hexdigest())
    here = root / '.scl-publication'; here.mkdir()
    (here / 'manifest.json').write_text(json.dumps({'pre': pre, 'post': post,
        'payload_sha256': m.PAYLOAD, 'patch_sha256': m.PATCH}))
    n = (len(payload) + 3) // 4
    for i in range(4): (here / f'part{i}').write_bytes(payload[n*i:n*(i+1)])
    run(root, 'add', '.scl-publication'); run(root, 'commit', '-qm', 'transport')
    return root

def test_exact_source_archived(repo):
    m.apply(repo)
    manifest = json.loads((repo / m.DEST / 'evidence/source_manifest.json').read_text())
    m.verify(repo, manifest['post'])
    assert (repo / m.DEST / 'evidence/scl_source.patch').is_file()

def test_corrupt_payload_stops_before_mutation(repo):
    (repo / '.scl-publication/part0').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='payload checksum'): m.apply(repo)
    assert (repo / 'src/test_0.txt').read_text() == 'old 0\n'

def test_wrong_base_stops_before_mutation(repo):
    (repo / 'src/test_0.txt').write_text('someone else modified this')
    with pytest.raises(RuntimeError, match='source hash'): m.apply(repo)
    assert (repo / 'src/test_1.txt').read_text() == 'old 1\n'

@pytest.mark.parametrize('name', ['../secret', '/tmp/secret', 'src/../secret', '.github/workflows/evil.yml'])
def test_unapproved_paths(tmp_path, name):
    with pytest.raises(RuntimeError): m.checked_path(tmp_path, name)

def test_no_unapproved_publication(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_REPOSITORY', 'other/repo')
    with pytest.raises(RuntimeError, match='not approved'): m.publish(tmp_path)

def test_failing_checks_remain_failing(repo, monkeypatch):
    m.apply(repo)
    monkeypatch.setenv('PYTEST_OUTCOME', 'failure')
    m.record(repo)
    status = json.loads((repo / m.DEST / 'SNAPSHOT_STATUS.json').read_text())
    assert status['checks']['pytest'] == 'failure'
    assert status['source_materialized'] and not status['production_approved']
