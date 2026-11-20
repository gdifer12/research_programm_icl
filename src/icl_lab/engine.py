from contextlib import nullcontext
from pathlib import Path
import copy
import json
import math
import os
import random
import signal
import time
import uuid
import numpy as np
import torch
import yaml
from . import config as configuration
from .devices import require_device, synchronize, memory_stats
from .runtime import project_path
from .tasks import make_task
from .storage import (append_json, atomic_bytes, code_hash, environment, load_npz, load_torch,
                      lock, save_npz, save_torch, sha, snapshot, write_json, clean)

SPLITS = ['probe_train', 'validation', 'id_test', 'test']


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def setup(config, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = require_device(device, config['train']['precision'])
    torch.set_num_threads(config['train']['threads'])
    torch.use_deterministic_algorithms(config['train']['deterministic'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(config['model_seed'])
    np.random.seed(config['model_seed'])
    torch.manual_seed(config['model_seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['model_seed'])
    elif device.type == 'mps':
        torch.mps.manual_seed(config['model_seed'])
    return device


def prepare(config, directory, task=None):
    directory = project_path(directory, 'bank directory')
    with lock(directory):
        manifest_path = project_path(directory / 'manifest.json', 'bank manifest')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['config_hash'] != configuration.digest(config):
                raise ValueError('Bank configuration differs; choose a new output directory')
            for name, entry in manifest['banks'].items():
                if sha(directory / entry['file']) != entry['sha256']:
                    raise ValueError(f'Corrupt bank: {name}')
            return manifest
        if any(p.name != '.writer.lock' for p in directory.iterdir()):
            raise ValueError(f'Incomplete bank directory: {directory}; preserve it and choose another path')
        task = task or make_task(config)
        save_torch(directory / 'task-state.pt', task.state_dict())
        banks = {}
        for i, split in enumerate(SPLITS):
            n = config['eval']['diagnostic_episodes'] if split == 'probe_train' else config['eval']['episodes']
            generator = torch.Generator().manual_seed(config['eval_seed'] + 100003 * (i + 1))
            batch = task.sample(n, generator, split=split)
            if 'episode_id' not in batch:
                batch['episode_id'] = torch.arange(n)
            save_npz(directory / f'{split}.npz', batch)
            banks[split] = {'file': f'{split}.npz', 'sha256': sha(directory / f'{split}.npz'),
                            'episodes': n, 'seed': config['eval_seed'] + 100003 * (i + 1)}
        manifest = {'schema_version': 1, 'config_hash': configuration.digest(config),
                    'experiment': config['experiment'], 'banks': banks,
                    'task_state_sha256': sha(directory / 'task-state.pt'),
                    'technical_smoke': config.get('technical_smoke', False)}
        write_json(manifest_path, manifest)
        return manifest


def read_bank(run, split):
    directory = project_path(Path(run) / 'banks', 'bank directory')
    manifest = json.loads(project_path(directory / 'manifest.json', 'bank manifest').read_text())
    entry = manifest['banks'][split]
    path = project_path(directory / entry['file'], 'bank file')
    if sha(path) != entry['sha256']:
        raise ValueError('Bank hash mismatch')
    return load_npz(path), entry['sha256']


def summarize(arrays):
    result = {}
    for key, tensor in arrays.items():
        values = tensor.double().cpu().numpy()
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        total = np.where(finite, values, 0).sum(axis=0)
        mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
        result[key] = {'mean': mean.tolist(), 'count': count.tolist()}
    if 'mse' in result and 'context_mean_mse' in result:
        numerator = np.asarray(result['mse']['mean'])
        denominator = np.asarray(result['context_mean_mse']['mean'])
        result['paper_r2'] = {'mean': (1 - numerator / np.maximum(denominator, 1e-30)).tolist(),
                              'definition': '1 - mean squared error / mean context-mean baseline error'}
    return result


def evaluate_model(task, model, batch, batch_size, device, baselines=False):
    model.eval()
    predictions, metrics = [], {}
    n = next(iter(batch.values())).shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            part = {k: v[start:start + batch_size] for k, v in batch.items()}
            gpu = move(part, device)
            pred = task.predict(model, gpu)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite model prediction')
            predictions.append(pred.detach().cpu())
            for key, value in task.metrics(pred, gpu).items():
                metrics.setdefault(key, []).append(value.detach().cpu())
    arrays = {key: torch.cat(parts) for key, parts in metrics.items()}
    pred = torch.cat(predictions)
    baseline_outputs = task.baseline_predictions(batch) if baselines else {}
    baseline_metrics = {name: task.metrics(value, batch) for name, value in baseline_outputs.items()}
    return pred, arrays, baseline_outputs, baseline_metrics


def checkpoint_path(run, value):
    directory = project_path(Path(run) / 'checkpoints', 'checkpoint directory')
    path = directory / (value + '.pt') if value in {'last', 'previous', 'initial', 'final'} else Path(value)
    path = project_path(path, 'checkpoint')
    if not path.is_file():
        candidate = project_path(directory / value, 'checkpoint')
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_run(run, checkpoint='last', device='cpu'):
    run = project_path(run, 'run directory')
    config = yaml.safe_load(project_path(run / 'config.yaml', 'run configuration').read_text())
    device = setup(config, device)
    task = make_task(config)
    path = checkpoint_path(run, checkpoint)
    saved = load_torch(path, full=True)
    if saved['config_hash'] != configuration.digest(config):
        raise ValueError('Checkpoint configuration mismatch')
    task.load_state_dict(saved['task_state'])
    model = task.build_model().to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return config, task, model, saved, path, device


def evaluate(run, checkpoint='last', split='test', device='cpu'):
    run = project_path(run, 'run directory')
    with lock(run):
        config, task, model, saved, path, device = load_run(run, checkpoint, device)
        bank, bank_hash = read_bank(run, split)
        key = configuration.digest({'checkpoint': sha(path), 'bank': bank_hash, 'code': code_hash(),
                                    'eval': config['eval'], 'device_type': device.type, 'precision': 'fp32'})[:20]
        output = project_path(run / 'evaluations' / f'{split}-{saved["step"]:08d}-{key}', 'evaluation directory')
        output.mkdir(parents=True, exist_ok=True)
        summary_path = project_path(output / 'summary.json', 'evaluation summary')
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        synchronize(device)
        start = time.perf_counter()
        pred, metrics, baseline_outputs, baseline_metrics = evaluate_model(task, model, bank, config['eval']['batch_size'], device, True)
        arrays = {'prediction': pred, 'episode_id': bank['episode_id']}
        arrays.update({'metric__' + key: value for key, value in metrics.items()})
        arrays.update({'baseline__' + key: value for key, value in baseline_outputs.items()})
        save_npz(output / 'predictions.npz', arrays)
        summary = {'step': saved['step'], 'attempt_id': saved['attempt_id'], 'split': split, 'checkpoint_sha256': sha(path), 'bank_sha256': bank_hash,
                   'code_hash': code_hash(), 'model': summarize(metrics),
                   'baselines': {key: summarize(value) for key, value in baseline_metrics.items()},
                   'elapsed_seconds': time.perf_counter() - start,
                   'technical_smoke': config.get('technical_smoke', False), 'predictions': str(output / 'predictions.npz')}
        # Prompt-level bootstrap keeps every k of one episode together.
        if config['experiment'] == 'garg':
            values = metrics['nmse'].numpy()
            rng = np.random.default_rng(config['eval_seed'] + 900001)
            means = np.stack([values[rng.integers(0, len(values), len(values))].mean(0)
                              for _ in range(config['eval']['bootstrap_samples'])])
            summary['nmse_bootstrap_90'] = np.quantile(means, [0.05, 0.95], axis=0).tolist()
        summary = clean(summary)
        write_json(output / 'summary.json', summary)
        return summary


def rng_state(generator, device='cpu'):
    device = torch.device(device)
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'mps': torch.mps.get_rng_state().cpu() if device.type == 'mps' else None,
            'train_generator': generator.get_state()}


def restore_rng(state, generator, device='cpu', same_backend=True):
    device = torch.device(device)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if same_backend and device.type == 'cuda':
        if not state.get('cuda'):
            raise ValueError('CUDA resume checkpoint is missing its device RNG state')
        torch.cuda.set_rng_state_all(state['cuda'])
    if same_backend and device.type == 'mps':
        if state.get('mps') is None:
            raise ValueError('MPS resume checkpoint is missing its device RNG state')
        torch.mps.set_rng_state(state['mps'].cpu())
    generator.set_state(state['train_generator'])


def train(config, run, device='cpu', resume=None, stop_after=None, allow_code_change=False, allow_device_change=False):
    run = project_path(run, 'run directory')
    with lock(run):
        return _train(config, run, device, resume, stop_after, allow_code_change, allow_device_change)


def _train(config, run, device, resume, stop_after, allow_code_change, allow_device_change):
    if resume:
        existing = yaml.safe_load(project_path(run / 'config.yaml', 'run configuration').read_text())
        if config is not None and configuration.digest(existing) != configuration.digest(config):
            raise ValueError('Resume requires the same resolved config (including total steps); use --stop-after for a pilot')
        config = existing
    elif any(p.name != '.writer.lock' for p in run.iterdir()):
        raise FileExistsError('Run directory is not empty; use --resume or a fresh directory')
    configuration.validate(config)
    device = setup(config, device)
    task = make_task(config)
    model = task.build_model().to(device)
    train_cfg = config['train']
    optimizer_cls = torch.optim.AdamW if train_cfg['optimizer'].lower() == 'adamw' else torch.optim.Adam
    optimizer = optimizer_cls([p for p in model.parameters() if p.requires_grad], lr=train_cfg['lr'],
                              betas=tuple(train_cfg['betas']), weight_decay=train_cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg['precision'] == 'fp16')
    generator = torch.Generator().manual_seed(config['train_seed'])
    config_hash = configuration.digest(config)
    attempt = uuid.uuid4().hex[:12]
    step, attempted, skipped = 0, 0, 0
    current_code = code_hash()
    prior_device, device_changed = None, False
    if resume:
        resume_path = checkpoint_path(run, resume)
        state = load_torch(resume_path, full=True)
        if not state.get('resumable') or state['config_hash'] != config_hash:
            raise ValueError('Checkpoint is not a matching full resume state')
        if state['code_hash'] != current_code and not allow_code_change:
            raise ValueError('Code changed since checkpoint; use --allow-code-change to record an explicit new attempt')
        prior_device = state.get('training_device')
        if prior_device is None:
            # Before native MPS support, the original device lived in manifest.json.
            prior_device = json.loads(project_path(run / 'manifest.json', 'run manifest').read_text())['environment']['device']
        device_changed = torch.device(prior_device).type != device.type
        if device_changed and not allow_device_change:
            raise ValueError(f'Resume backend changed from {prior_device} to {device}; use --allow-device-change to record a non-identical continuation')
        task.load_state_dict(state['task_state'])
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        step, attempted, skipped = state['step'], state['attempted'], state['skipped']
        restore_rng(state['rng'], generator, device, same_backend=not device_changed)
        abandoned = project_path(run / 'checkpoints' / 'abandoned' / attempt, 'abandoned checkpoints')
        def archive_old(old_path, reason):
            old_path = project_path(old_path, 'checkpoint to archive')
            abandoned.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, project_path(abandoned / old_path.name, 'archived checkpoint'))
            checksum = project_path(old_path.with_suffix('.pt.sha256'), 'checkpoint checksum')
            if checksum.exists():
                os.replace(checksum, project_path(abandoned / checksum.name, 'archived checksum'))
            append_json(run / 'events.jsonl', {'event':'archive_checkpoint', 'path':str(old_path), 'reason':reason})
        for old_path in list((run / 'checkpoints').glob('step-*.pt')) + [run / 'checkpoints' / 'final.pt']:
            old_path = project_path(old_path, 'optional checkpoint')
            if not old_path.is_file():
                continue
            try:
                old_step = load_torch(old_path, full=True)['step']
            except Exception as error:
                archive_old(old_path, 'unreadable optional snapshot: ' + repr(error))
                continue
            if old_step > step:
                archive_old(old_path, 'abandoned after rollback')
        last_path = project_path(run / 'checkpoints' / 'last.pt', 'last checkpoint')
        if resume_path.resolve() != last_path.resolve():
            # The first new checkpoint must rotate from the selected resume state,
            # never from a valid but abandoned future last checkpoint.
            if last_path.exists():
                archive_old(last_path, 'replaced by selected resume state')
            save_torch(last_path, state)
    else:
        atomic_bytes(run / 'config.yaml', yaml.safe_dump(config, sort_keys=False).encode())
        snapshot(run / 'provenance')
        write_json(run / 'manifest.json', {'schema_version': 1, 'config_hash': config_hash,
                   'code_hash': current_code, 'environment': environment(device),
                   'parameter_count': sum(p.numel() for p in model.parameters()),
                   'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'technical_smoke': config.get('technical_smoke', False)})
    if resume:
        snapshot(run / 'attempts' / attempt)
        write_json(run / 'attempts' / attempt / 'environment.json', environment(device))
    before_banks = rng_state(generator, device)
    prepare(config, run / 'banks', task)
    # Evaluation bank preparation is isolated from every training RNG.
    restore_rng(before_banks, generator, device)
    val, _ = read_bank(run, 'validation')
    checkpoint_dir = project_path(run / 'checkpoints', 'checkpoint directory')
    checkpoint_dir.mkdir(exist_ok=True)
    start_step = step
    append_json(run / 'events.jsonl', {'event': 'start', 'attempt_id': attempt, 'resume_from_step': step,
                                      'code_hash': current_code, 'time': time.time(), 'training_device': str(device),
                                      'prior_training_device': prior_device, 'device_changed': device_changed})
    stop = {'requested': False}
    old_handlers = {}
    def request_stop(signum, frame):
        stop['requested'] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, request_stop)
    selected = set(np.unique(np.geomspace(1, train_cfg['steps'], train_cfg['analysis_checkpoints']).astype(int)))
    if train_cfg['analysis_every']:
        selected.update(range(train_cfg['analysis_every'], train_cfg['steps'] + 1, train_cfg['analysis_every']))
    selected.add(train_cfg['steps'])

    def payload(full):
        result = {'schema_version': 1, 'step': step, 'model': model.state_dict(), 'task_state': task.state_dict(),
                  'config_hash': config_hash, 'code_hash': current_code, 'resumable': full,
                  'attempt_id': attempt, 'training_device': str(device)}
        if full:
            result.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng_state(generator, device),
                          attempted=attempted, skipped=skipped)
        return result

    last_saved_step = step if resume and not device_changed else -1

    def save_last():
        nonlocal last_saved_step
        if last_saved_step == step:
            return
        before = time.perf_counter()
        last = project_path(checkpoint_dir / 'last.pt', 'last checkpoint')
        previous = project_path(checkpoint_dir / 'previous.pt', 'previous checkpoint')
        # Write the new checkpoint first; keep last intact if disk space is exhausted.
        pending = project_path(checkpoint_dir / 'pending.pt', 'pending checkpoint')
        save_torch(pending, payload(True))
        last_checksum = project_path(last.with_suffix('.pt.sha256'), 'last checkpoint checksum')
        last_valid = last.is_file() and last_checksum.is_file() and sha(last) == last_checksum.read_text().strip()
        if last_valid:
            # Keep last readable while preparing previous. At every crash boundary
            # at least one of last/previous retains a complete verified pair.
            backup = project_path(checkpoint_dir / 'previous.next.pt', 'checkpoint backup')
            if backup.exists():
                backup.unlink()
            os.link(last, backup)
            os.replace(backup, previous)
            atomic_bytes(previous.with_suffix('.pt.sha256'), last.with_suffix('.pt.sha256').read_bytes())
        os.replace(pending, last)
        os.replace(pending.with_suffix('.pt.sha256'), last.with_suffix('.pt.sha256'))
        last_saved_step = step
        append_json(run / 'events.jsonl', {'event': 'checkpoint', 'step': step, 'attempt_id': attempt,
                                          'seconds': time.perf_counter()-before, 'bytes': last.stat().st_size})

    if not resume:
        save_torch(checkpoint_dir / 'initial.pt', payload(False))
        save_last()
    write_json(run / 'status.json', {'state': 'running', 'step': step, 'attempt_id': attempt,
                                    'technical_smoke': config.get('technical_smoke', False)})
    target = min(train_cfg['steps'], step + stop_after) if stop_after is not None else train_cfg['steps']
    elapsed_training = 0.0
    attempt_start = time.perf_counter()
    try:
        while step < target and not stop['requested']:
            model.train()
            progress = step / max(train_cfg['steps'] - 1, 1)
            factor = 0.5 * (1 + math.cos(math.pi * progress)) if train_cfg['scheduler'] == 'cosine' else 1.0
            if step < train_cfg['warmup_steps']:
                factor = (step + 1) / train_cfg['warmup_steps']
            for group in optimizer.param_groups:
                group['lr'] = train_cfg['lr'] * factor
            synchronize(device)
            before = time.perf_counter()
            batch = move(task.sample(train_cfg['batch_size'], generator, 'train', step), device)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast('cuda', dtype=torch.float16) if scaler.is_enabled() else nullcontext()
            with autocast:
                pred = task.predict(model, batch)
                loss = task.loss(pred, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss after {step} completed updates')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'] or float('inf'))
            attempted += 1
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite FP32 gradient')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            was_skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
            synchronize(device)
            duration = time.perf_counter() - before
            elapsed_training += duration
            if was_skipped:
                skipped += 1
                append_json(run / 'events.jsonl', {'event': 'amp_skip', 'attempt_id': attempt, 'step': step, 'scale': scaler.get_scale()})
                if skipped > max(100, attempted // 2):
                    raise FloatingPointError('Too many skipped AMP updates')
                continue
            step += 1
            if step % train_cfg['log_every'] == 0 or step == target:
                row = {'step': step, 'attempt_id': attempt, 'train_loss': loss.item(), 'lr': optimizer.param_groups[0]['lr'],
                       'grad_norm': norm.item(), 'update_seconds': duration, 'episodes_per_second': train_cfg['batch_size']/duration,
                       'attempted_batches': attempted, 'skipped_updates': skipped, 'amp_scale': scaler.get_scale()}
                row.update(device=str(device), **memory_stats(device))
                append_json(run / 'metrics.jsonl', row)
                print(json.dumps(row), flush=True)
            if step % train_cfg['eval_every'] == 0 or step == target:
                before = time.perf_counter()
                _, measures, _, _ = evaluate_model(task, model, val, config['eval']['batch_size'], device)
                append_json(run / 'validation.jsonl', {'step': step, 'attempt_id': attempt, 'metrics': summarize(measures),
                                                       'seconds': time.perf_counter()-before})
            if step in selected:
                save_torch(checkpoint_dir / f'step-{step:08d}.pt', payload(False))
            if step % train_cfg['save_every'] == 0:
                save_last()
        save_last()
        complete = step >= train_cfg['steps']
        if complete:
            save_torch(checkpoint_dir / 'final.pt', payload(False))
        status = {'state': 'completed' if complete else 'paused', 'step': step, 'attempt_id': attempt,
                  'resume_from_step': start_step, 'training_seconds_this_attempt': elapsed_training,
                  'loop_wall_seconds_this_attempt': time.perf_counter() - attempt_start,
                  'stop_requested': stop['requested'], 'technical_smoke': config.get('technical_smoke', False)}
        write_json(run / 'status.json', status)
        append_json(run / 'events.jsonl', dict(status, event='finish'))
        return status
    except BaseException as error:
        write_json(run / 'status.json', {'state': 'failed', 'step': step, 'attempt_id': attempt,
                                        'error': repr(error), 'resume': 'last or previous full checkpoint'})
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
