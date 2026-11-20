"""Launch the two Kobayashi series from a notebook; standard library only."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

from icl_lab.runtime import ROOT, project_path, require_local_environment

BASE = ROOT / 'artifacts/runs/kobayashi-jupyter-20261119'
CONFIGS = {'vanilla': 'kobayashi.yaml', 'hypernetwork': 'kobayashi_hyper.yaml'}
STEPS = 100000


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    path = project_path(path, 'series JSON')
    return json.loads(path.read_text()) if path.is_file() else {}


def sha(path):
    with project_path(path, 'hashed file', must_exist=True).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = project_path(path, 'series JSON output')
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_root(variant, run_root=None):
    if variant not in CONFIGS:
        raise ValueError('variant must be vanilla or hypernetwork')
    root = Path(run_root) if run_root is not None else BASE / variant
    root = project_path(root if root.is_absolute() else ROOT / root, 'series root')
    if root.name != variant:
        raise ValueError('run_root must end with the variant name; variants share one parent')
    return root


def identity(variant, seeds):
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct non-negative integers')
    config = project_path(ROOT / 'configs/experiments' / CONFIGS[variant], must_exist=True)
    # Same source digest as icl_lab.storage.code_hash(), without importing torch
    # into the separate Jupyter environment.
    files = []
    source = project_path(ROOT / 'src', 'source directory', must_exist=True)
    for candidate in source.rglob('*'):
        if '__pycache__' in candidate.parts:
            continue
        path = project_path(candidate, 'source file')
        if path.is_file():
            files.append(path)
    digest = hashlib.sha256(''.join(str(path.relative_to(ROOT)) + sha(path)
                                   for path in sorted(files)).encode()).hexdigest()
    return {'schema_version': 1, 'experiment': 'kobayashi', 'variant': variant, 'seeds': seeds,
            'device': 'mps', 'steps': STEPS, 'code_hash': digest,
            'config_source_sha256': sha(config)}


def check_identity(root, expected):
    saved = read_json(root / 'series-manifest.json')
    if saved and any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Series configuration/code/seeds changed: use a new series parent directory')
    return saved


def final_evaluations_exist(run, expected):
    status = read_json(run / 'status.json')
    if status.get('state') != 'completed' or status.get('step') != expected['steps']:
        return False
    directory = project_path(run / 'evaluations', 'evaluations directory')
    checkpoint = project_path(run / 'checkpoints' / f'step-{expected["steps"]:08d}.pt', 'final snapshot')
    if not directory.is_dir() or not checkpoint.is_file():
        return False
    checkpoint_hash = sha(checkpoint)
    banks = read_json(run / 'banks/manifest.json').get('banks', {})
    for split in ['id_test', 'test']:
        bank = banks.get(split, {})
        if not bank.get('file') or not bank.get('sha256'):
            return False
        bank_path = project_path(run / 'banks' / bank['file'], 'evaluation bank')
        if not bank_path.is_file() or sha(bank_path) != bank['sha256']:
            return False
        found = False
        for candidate in directory.iterdir():
            if not candidate.name.startswith(split + '-'):
                continue
            entry = project_path(candidate, 'evaluation output directory')
            if not entry.is_dir():
                continue
            path = project_path(entry / 'summary.json', 'evaluation summary')
            summary = read_json(path)
            if (summary.get('step') == expected['steps'] and summary.get('split') == split
                    and summary.get('code_hash') == expected['code_hash']
                    and summary.get('checkpoint_sha256') == checkpoint_hash
                    and summary.get('bank_sha256') == bank['sha256']
                    and summary.get('predictions')):
                found = project_path(path.with_name('predictions.npz'), 'evaluation predictions').is_file()
                if found:
                    break
        if not found:
            return False
    return True


def completed(root, expected):
    return (bool(check_identity(root, expected))
            and read_json(root / 'series-state.json').get('state') == 'completed'
            and all(final_evaluations_exist(root / f'seed-{seed}', expected) for seed in expected['seeds']))


def check_dependency(root, expected):
    if expected['variant'] == 'hypernetwork':
        vanilla = project_path(root.parent / 'vanilla', 'vanilla dependency')
        if not completed(vanilla, identity('vanilla', expected['seeds'])):
            raise RuntimeError('Hypernetwork requires the completed vanilla series with final ID/OOD evaluations')


def launch_series(variant, seeds=(1, 2, 3), run_root=None):
    """Return launch/status information without waiting or polling GPU progress.

    run_root is the variant directory, not the shared parent. Existing completed
    series are reused; incomplete series resume through the detached worker.
    """
    require_local_environment()
    root = resolve_root(variant, run_root)
    root.mkdir(parents=True, exist_ok=True)
    expected = identity(variant, seeds)
    check_identity(root, expected)
    log_path = project_path(root / 'worker.log', 'worker log')
    info = {'variant': variant, 'seeds': expected['seeds'], 'run_root': str(root), 'log': str(log_path)}
    with project_path(root / '.writer.lock', 'series lock').open('a+') as writer:
        try:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**read_json(root / 'series-state.json'), **info, 'state': 'running', 'already_running': True}
        if completed(root, expected):
            return {**read_json(root / 'series-state.json'), **info, 'already_completed': True}
        with project_path(root.parent / '.gpu-series.lock', 'shared GPU lock').open('a+') as gpu:
            try:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {**info, 'state': 'blocked', 'pid': None, 'reason': 'Another Kobayashi variant or its child is running'}
            check_dependency(root, expected)
            previous = read_json(root / 'series-state.json')
            pending = {**expected, 'state': 'starting', 'pid': None, 'stage': 'launch',
                       'started_at': previous.get('started_at', now()), 'worker_started_at': now(),
                       'stages': previous.get('stages', [])}
            write_json(root / 'series-state.json', pending)
            command = [str(project_path(ROOT / 'scripts/local-python', must_exist=True)),
                       str(project_path(ROOT / 'scripts/run-kobayashi-series.py', must_exist=True)),
                       '--variant', variant, '--run-root', str(root), '--seeds', *map(str, expected['seeds']),
                       '--lock-fd', str(writer.fileno()), '--gpu-lock-fd', str(gpu.fileno())]
            try:
                with log_path.open('a') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                             pass_fds=(writer.fileno(), gpu.fileno()))
            except BaseException as error:
                write_json(root / 'series-state.json', {**pending, 'state': 'failed', 'error': str(error)})
                raise
            # Do not explicitly unlock: inherited descriptors protect startup
            # and remain held by worker/child after this notebook call returns.
            warning = None
            try:
                if caffeinate := shutil.which('caffeinate'):
                    subprocess.Popen([caffeinate, '-i', '-w', str(child.pid)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
            except OSError as error:
                warning = f'Worker started, but sleep prevention failed: {error}'
            return {**info, 'state': 'running', 'pid': child.pid, 'worker_started_at': pending['worker_started_at'],
                    'warning': warning}
