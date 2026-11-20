"""Fork the completed Assign and Add seed 2 at 30k for native resume to 60k."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
import yaml

from icl_lab import config
from icl_lab.engine import SPLITS
from icl_lab.runtime import ROOT, project_path
from icl_lab.storage import atomic_bytes, code_hash, load_torch, save_torch, sha, write_json
from icl_lab.tasks import make_task


SOURCE = ROOT / 'artifacts/runs/assign-add-jupyter-20261119/seed-2'
SOURCE_SHA = 'd67835c26c55a0614d2d77f4fbad6a7b114d8a0e6da6cb4b3c1c5cf401c99dbd'
START, TARGET = 30000, 60000


def identical(left, right, key='state'):
    """Compare serialization round trips, including NumPy and Torch RNG states."""
    if type(left) is not type(right):
        raise ValueError(f'{key}: differing types')
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f'{key}: differing keys')
        for name in left:
            identical(left[name], right[name], f'{key}.{name}')
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError(f'{key}: differing lengths')
        for i, (a, b) in enumerate(zip(left, right)):
            identical(a, b, f'{key}[{i}]')
    elif isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f'{key}: different tensor')
    elif isinstance(left, np.ndarray):
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError(f'{key}: different array')
    elif left != right:
        raise ValueError(f'{key}: differing values')


def read_json(path):
    return json.loads(path.read_text())


def verify_parent(parent_config, manifest, state):
    expected = config.load(ROOT / 'configs/experiments/assign_add.yaml', seed=2)
    identical(expected, parent_config, 'source config')
    train = parent_config['train']
    if (train['steps'], train['scheduler'], train['warmup_steps'], train['precision'],
            parent_config['train_seed']) != (START, 'constant', 0, 'fp32', 10002):
        raise ValueError('The original run must use a constant, warmup-free 30k schedule')
    status = read_json(SOURCE / 'status.json')
    if (status['state'], status['step'], status['attempt_id']) != ('completed', START, state['attempt_id']):
        raise ValueError('Parent status and checkpoint disagree')
    if (state['resumable'], state['step'], state['attempted'], state['skipped'],
            state['training_device']) != (True, START, START, 0, 'mps'):
        raise ValueError('Expected a full, completed MPS checkpoint with zero skipped updates')
    if not (config.digest(parent_config) == state['config_hash'] == manifest['config_hash']
            and code_hash() == state['code_hash'] == manifest['code_hash']):
        raise ValueError('Parent config or scientific source digest mismatch')
    final = load_torch(SOURCE / 'checkpoints/final.pt', full=True)
    identical(state['model'], final['model'], 'parent final weights')
    task = make_task(parent_config)
    identical(state['task_state'], task.state_dict(), 'task state')
    model = task.build_model()
    model.load_state_dict(state['model'])
    opt = state['optimizer']
    group, = opt['param_groups']
    if (group['lr'], group['betas'], group['weight_decay']) != (
            train['lr'], tuple(train['betas']), train['weight_decay']):
        raise ValueError('AdamW hyperparameters differ from the source config')
    for param_id, param in zip(group['params'], model.parameters(), strict=True):
        moments = opt['state'][param_id]
        if moments['step'].item() != START or not torch.isfinite(param).all():
            raise ValueError('Invalid optimizer step or nonfinite weights')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = moments[name]
            if value.shape != param.shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid AdamW {name}')
    if state['scaler'] != {} or state['rng']['mps'] is None:
        raise ValueError('Expected disabled FP32 scaler and saved Metal RNG')
    # Assign and Add consumes exactly one CPU generator draw per sampled batch.
    stream = torch.Generator().manual_seed(parent_config['train_seed'])
    for _ in range(START):
        torch.randint(0, 2**63 - 1, (), generator=stream)
    identical(stream.get_state(), state['rng']['train_generator'], 'training stream after 30k')
    batch = task.sample(train['batch_size'], stream, 'train', START)
    return hashlib.sha256(batch['tokens'].numpy().tobytes() + batch['target'].numpy().tobytes()).hexdigest()


def copy_banks(destination, source_config_hash, fork_config_hash):
    banks = SOURCE / 'banks'
    descriptor = read_json(banks / 'manifest.json')
    if descriptor['config_hash'] != source_config_hash or set(descriptor['banks']) != set(SPLITS):
        raise ValueError('Parent bank manifest does not match the run')
    destination.mkdir()
    files = {'task-state.pt': descriptor['task_state_sha256']}
    for split, entry in descriptor['banks'].items():
        if entry['file'] != f'{split}.npz':
            raise ValueError(f'Unexpected parent bank filename for {split}')
        files[entry['file']] = entry['sha256']
    for name, digest in files.items():
        shutil.copyfile(banks / name, destination / name)
        if sha(destination / name) != digest:
            raise ValueError(f'Parent bank checksum mismatch: {name}')
    atomic_bytes(destination / 'task-state.pt.sha256', (files['task-state.pt'] + '\n').encode())
    descriptor['config_hash'] = fork_config_hash
    write_json(destination / 'manifest.json', descriptor)
    return files


def prepare(run_dir):
    run = project_path(run_dir, 'new continuation directory')
    if not any(base in run.parents for base in (ROOT / 'artifacts/runs', ROOT / 'artifacts/tmp')):
        raise ValueError('Place the new run below artifacts/runs or artifacts/tmp')
    if run.is_relative_to(SOURCE) or SOURCE.is_relative_to(run):
        raise ValueError('The scientific parent run must remain read-only')
    if run.exists():
        raise FileExistsError(f'Continuation destination already exists: {run}')
    torch.set_num_threads(1)
    parent_config = yaml.safe_load((SOURCE / 'config.yaml').read_text())
    manifest = read_json(SOURCE / 'manifest.json')
    source_checkpoint = SOURCE / 'checkpoints/last.pt'
    if sha(source_checkpoint) != SOURCE_SHA:
        raise ValueError('The original full checkpoint does not have the recorded digest')
    state = load_torch(source_checkpoint, full=True)
    batch_hash = verify_parent(parent_config, manifest, state)
    fork_config = config.merge(parent_config, {'train': {'steps': TARGET}})
    config.validate(fork_config)
    fork_hash = config.digest(fork_config)
    run.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run.parent, prefix='.assign-continuation-') as tmp:
        stage = Path(tmp) / 'run'
        stage.mkdir()
        atomic_bytes(stage / 'config.yaml', yaml.safe_dump(fork_config, sort_keys=False).encode())
        files = copy_banks(stage / 'banks', state['config_hash'], fork_hash)
        identical(state['task_state'], load_torch(stage / 'banks/task-state.pt'), 'bank task state')
        anchor = stage / 'checkpoints/parent-00030000.pt'
        # Only the metadata digest changes. Every field is validated after serialization.
        save_torch(anchor, dict(state, config_hash=fork_hash))
        restored = load_torch(anchor, full=True)
        if restored['config_hash'] != fork_hash:
            raise ValueError('Fork checkpoint has incorrect config digest')
        restored['config_hash'] = state['config_hash']
        identical(state, restored, 'full fork state')
        for suffix in ('.pt', '.pt.sha256'):
            shutil.copyfile(anchor.with_suffix(suffix), anchor.parent / ('last' + suffix))
        if sha(anchor.parent / 'last.pt') != sha(anchor):
            raise ValueError('Fork last checkpoint differs from preserved anchor')
        provenance = {
            'parent_run': str(SOURCE.relative_to(ROOT)), 'parent_checkpoint': 'checkpoints/last.pt',
            'source_sha256': SOURCE_SHA, 'fork_anchor': 'checkpoints/' + anchor.name,
            'fork_anchor_sha256': sha(anchor), 'source_config_hash': state['config_hash'],
            'child_config_hash': fork_hash, 'code_hash': state['code_hash'],
            'config_delta': {'train.steps': [START, TARGET]}, 'model_seed': 2, 'train_seed': 10002,
            'parent_attempt_id': state['attempt_id'], 'banks_sha256': files,
            'training_stream': 'One CPU generator draw per batch verified through step 30000.',
            'next_batch_tokens_then_targets_sha256': batch_hash,
            'state_check': 'All fields of the full checkpoint exact after round trip except config_hash.',
            'preparer_sha256': sha(Path(__file__)),
        }
        write_json(stage / 'provenance/continuation.json', provenance)
        write_json(stage / 'manifest.json', dict(manifest, config_hash=fork_hash,
                   parent_run=provenance['parent_run'], environment_origin='parent manifest',
                   continuation='provenance/continuation.json'))
        write_json(stage / 'status.json', {'state': 'paused', 'step': START, 'technical_smoke': False,
                   'attempt_id': state['attempt_id'], 'reason': 'prepared independent continuation'})
        # Concurrent preparations publish at most one complete directory.
        fd = os.open(run.parent, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if run.exists():
                raise FileExistsError(run)
            stage.rename(run)
        finally:
            os.close(fd)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().run_dir), indent=2))
