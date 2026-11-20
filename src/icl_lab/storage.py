from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import tarfile
import tempfile
import time
import numpy as np
import torch
from .runtime import ROOT, project_path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def sha(path):
    path = project_path(path, 'hash input', must_exist=True)
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = project_path(path, 'output')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, obj):
    atomic_bytes(path, (json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def append_json(path, obj):
    path = project_path(path, 'log output')
    with open(path, 'a') as stream:
        stream.write(json.dumps(clean(obj), ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def save_torch(path, obj):
    path = project_path(path, 'checkpoint output')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        with open(tmp, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        atomic_bytes(path.with_suffix(path.suffix + '.sha256'), (sha(path) + '\n').encode())
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_torch(path, full=False):
    path = project_path(path, 'checkpoint input')
    checksum = project_path(path.with_suffix(path.suffix + '.sha256'), 'checkpoint checksum')
    if not checksum.is_file() or sha(path) != checksum.read_text().strip():
        raise ValueError(f'Missing or invalid checkpoint checksum: {path}')
    # full=True is restricted to this program's trusted local resume files.
    return torch.load(path, map_location='cpu', weights_only=not full)


def save_npz(path, arrays):
    path = project_path(path, 'array output')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **{k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                                          for k, v in arrays.items()})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_npz(path):
    path = project_path(path, 'array input', must_exist=True)
    with np.load(path, allow_pickle=False) as data:
        return {k: torch.from_numpy(data[k].copy()) for k in data.files}


@contextmanager
def lock(directory):
    directory = project_path(directory, 'lock directory')
    directory.mkdir(parents=True, exist_ok=True)
    with open(project_path(directory / '.writer.lock', 'lock file'), 'a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is writing {directory}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_files():
    files = []
    for directory in ['src', 'configs', 'requirements', 'scripts', 'tests']:
        source = project_path(ROOT / directory, 'snapshot source directory', must_exist=True)
        for candidate in source.rglob('*'):
            if '__pycache__' in candidate.parts:
                continue
            checked = project_path(candidate, 'snapshot source')
            if checked.is_file():
                files.append(checked)
    files += [ROOT / p for p in ['pyproject.toml', 'environment.yml'] if (ROOT / p).is_file()]
    return sorted(project_path(p, 'snapshot source', must_exist=True) for p in files)


def code_hash():
    # Environment/docs/config changes must not invalidate a model-only resume.
    return hashlib.sha256(''.join(str(p.relative_to(ROOT)) + sha(p)
                                 for p in source_files() if 'src' in p.relative_to(ROOT).parts[:1]).encode()).hexdigest()


def snapshot(directory):
    directory = project_path(directory, 'source snapshot')
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files()}
    with tarfile.open(project_path(directory / 'source.tar.gz', 'snapshot archive'), 'w:gz') as archive:
        for rel in hashes:
            archive.add(ROOT / rel, arcname=rel)
    git = {}
    git_env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    for name, args in [('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--short']),
                       ('diff', ['diff', '--no-ext-diff', 'HEAD', '--', '.'])]:
        try:
            result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, env=git_env)
        except FileNotFoundError:
            git[name] = None
            continue
        git[name] = result.stdout if result.returncode == 0 else None
    write_json(directory / 'source.json', {'files': hashes, 'archive_sha256': sha(directory / 'source.tar.gz'),
                                          'code_hash': code_hash(), 'git': git})


def environment(device):
    from .devices import device_info
    freeze = subprocess.run([os.sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True)
    data = {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'device': str(device), 'pip_freeze': freeze.stdout.splitlines(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'job_id': os.environ.get('SLURM_JOB_ID') or os.environ.get('PBS_JOBID'),
            'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if torch.device(device).type == 'cuda':
        data.update(gpu=torch.cuda.get_device_name(device), capability=torch.cuda.get_device_capability(device),
                    cuda_arches=torch.cuda.get_arch_list())
    data.update(device_info(device))
    return data
