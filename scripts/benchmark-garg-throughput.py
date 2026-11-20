#!/usr/bin/env python3
"""Bounded technical Garg timing; use scripts/local-python, never as a science run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

from icl_lab.runtime import ROOT, project_path, require_local_environment
import torch
from icl_lab import config as configuration, engine
from icl_lab.devices import synchronize
from icl_lab.storage import code_hash, environment, sha, write_json
from icl_lab.tasks import make_task

BASE = ROOT / 'artifacts/validation/garg-throughput-20261119'
MODES = ['baseline', 'thread_prefetch', 'precomputed']


def stats(values):
    return {'samples': values, 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)} if values else {'samples': [], 'mean': None}


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(f'{name}:{tensor.dtype}:{list(tensor.shape)}:'.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Utilization:
    """Best-effort IOAccelerator samples; an absent counter is never interpreted as zero."""
    def __init__(self):
        self.samples, self.errors = [], []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.started = None

    def start(self):
        self.started = time.perf_counter()
        self.thread.start()

    def collect(self):
        while not self.done.is_set():
            try:
                output = subprocess.run(['/usr/sbin/ioreg', '-r', '-c', 'IOAccelerator', '-d1', '-w0'],
                                        capture_output=True, text=True, timeout=2, check=True).stdout
                values = [float(value) for value in re.findall(r'"Device Utilization(?: %)?"\s*=\s*([0-9.]+)', output)]
                self.samples.append({'elapsed_seconds': time.perf_counter() - self.started,
                                     'device_utilization_percent': values or None})
                if not values and 'Device Utilization counter unavailable' not in self.errors:
                    self.errors.append('Device Utilization counter unavailable')
            except (OSError, subprocess.SubprocessError) as error:
                self.errors.append(repr(error))
            self.done.wait(1.0)

    def stop(self):
        self.done.set()
        if self.started is not None:
            self.thread.join(3)
        values = [value for sample in self.samples for value in sample['device_utilization_percent'] or []]
        return {'state': 'available' if values else 'unavailable', 'samples': self.samples,
                'mean_reported_percent': statistics.mean(values) if values else None,
                'errors': self.errors, 'interval_seconds': 1.0,
                'meaning': 'IOAccelerator Device Utilization %, all reported accelerator counters; not kernel occupancy'}


def build(config):
    device = engine.setup(config, 'mps')  # Resets model, NumPy, Python and MPS seeds each time.
    task = make_task(config)
    model = task.build_model()
    initial_hash = tensor_hash(model.state_dict())  # CPU weights, outside timing.
    model = model.to(device)
    train = config['train']
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=train['lr'], betas=tuple(train['betas']), weight_decay=train['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return device, task, model, optimizer, scaler, initial_hash


def update(next_batch, task, model, optimizer, scaler, device, parts=None):
    def stage(name, operation):
        if parts is None:
            return operation()
        synchronize(device)
        wall, cpu = time.perf_counter(), time.process_time()
        result = operation()
        synchronize(device)
        parts.setdefault(name, []).append({'wall_seconds': time.perf_counter() - wall,
                                          'process_seconds': time.process_time() - cpu})
        return result

    batch = stage('sample', next_batch)
    batch = stage('transfer', lambda: engine.move(batch, device))

    def forward():
        optimizer.zero_grad(set_to_none=True)
        return task.predict(model, batch)

    prediction = stage('forward_and_zero_grad', forward)

    def loss_finite():
        loss = task.loss(prediction, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite benchmark loss')
        return loss

    loss = stage('loss_and_finite_check', loss_finite)

    def backward():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

    stage('backward', backward)

    def gradnorm():
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite benchmark gradient')

    stage('gradnorm_clip_inf_and_finite_check', gradnorm)

    def step():
        scaler.step(optimizer)
        scaler.update()

    stage('optimizer', step)
    return loss.detach()


def run_case(config, args, mode):
    device, task, model, optimizer, scaler, weights_hash = build(config)
    generator = torch.Generator().manual_seed(config['train_seed'])
    sample = lambda: task.sample(config['train']['batch_size'], generator, 'train', args.sample_step)
    total = args.warmup + args.timed
    pool, future, batches = None, None, None
    monitor = Utilization()
    wall_samples, cpu_samples, first_batch, last_batch = [], [], None, None
    setup_start, setup_cpu = time.perf_counter(), time.process_time()
    try:
        if mode == 'thread_prefetch':
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(sample)
        elif mode == 'precomputed':
            batches = [sample() for _ in range(total)]
        setup_wall, setup_process = time.perf_counter() - setup_start, time.process_time() - setup_cpu

        def next_batch():
            nonlocal future, first_batch, last_batch
            if mode == 'thread_prefetch':
                batch = future.result()
                if index + 1 < total:  # Exactly one pending batch, no discarded extra RNG draw.
                    future = pool.submit(sample)
            else:
                batch = batches[index] if batches is not None else sample()
            if index == args.warmup:
                first_batch = batch
            if index == total - 1:
                last_batch = batch
            return batch

        for index in range(total):
            model.train()
            synchronize(device)
            if index == args.warmup:
                monitor.start()
                total_wall, total_cpu = time.perf_counter(), time.process_time()
            wall, cpu = time.perf_counter(), time.process_time()
            loss = update(next_batch, task, model, optimizer, scaler, device)
            synchronize(device)
            elapsed, process = time.perf_counter() - wall, time.process_time() - cpu
            if index >= args.warmup:
                wall_samples.append(elapsed)
                cpu_samples.append(process)
                if index == args.warmup:
                    first_loss = loss
        total_wall, total_cpu = time.perf_counter() - total_wall, time.process_time() - total_cpu
        telemetry = monitor.stop()
        return {'mode': mode, 'wall_seconds': stats(wall_samples), 'process_seconds': stats(cpu_samples),
                'block_means_wall_seconds': [statistics.mean(wall_samples[i:i + 30]) for i in range(0, args.timed, 30)],
                'timed_phase_wall_seconds': total_wall, 'timed_phase_process_seconds': total_cpu,
                'input_setup_wall_seconds': setup_wall, 'input_setup_process_seconds': setup_process,
                'initial_weights_sha256': weights_hash, 'first_timed_batch_sha256': tensor_hash(first_batch),
                'last_timed_batch_sha256': tensor_hash(last_batch),
                'generator_after_sha256': tensor_hash({'state': generator.get_state()}),
                'first_timed_loss': first_loss.item(), 'last_timed_loss': loss.item(),
                'batch_shapes': {key: list(value.shape) for key, value in first_batch.items()},
                'utilization': telemetry}
    finally:
        monitor.stop()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def profiles(config, args):
    device, task, model, optimizer, scaler, _ = build(config)
    synchronize(device)
    generator = torch.Generator().manual_seed(config['train_seed'])
    sample = lambda: task.sample(config['train']['batch_size'], generator, 'train', args.sample_step)
    walls, cpus = [], []
    for _ in range(200):
        wall, cpu = time.perf_counter(), time.process_time()
        sample()
        walls.append(time.perf_counter() - wall)
        cpus.append(time.process_time() - cpu)
    generator.manual_seed(config['train_seed'])
    for _ in range(args.warmup):
        model.train()
        synchronize(device)
        update(sample, task, model, optimizer, scaler, device)
        synchronize(device)
    parts = {}
    for _ in range(10):
        model.train()
        update(sample, task, model, optimizer, scaler, device, parts)
    return {'cpu_sample_200': {'wall_seconds': stats(walls), 'process_seconds': stats(cpus)},
            'components_10': {'intrusive': True, 'method': 'Synchronize before and after every component; not additive throughput attribution',
                              'components': {name: {'wall_seconds': stats([v['wall_seconds'] for v in rows]),
                                                     'process_seconds': stats([v['process_seconds'] for v in rows])}
                                             for name, rows in parts.items()}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--timed', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--sample-step', type=int, default=30000)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--output', type=Path, help='New result JSON or directory inside the benchmark artifact directory')
    args = parser.parse_args()
    if not (0 <= args.warmup <= 200 and 1 <= args.timed <= 1000 and 1 <= args.repeats <= 6
            and 0 <= args.sample_step <= 500000 and len(set(args.modes)) == len(args.modes)):
        parser.error('Require warmup 0..200, timed 1..1000, repeats 1..6, sample-step 0..500000 and unique modes')
    require_local_environment()
    base = project_path(BASE, 'benchmark directory')
    base.mkdir(parents=True, exist_ok=True)
    destination = args.output or base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    destination = project_path(destination if destination.is_absolute() else ROOT / destination, 'benchmark output')
    result_path = project_path(destination if destination.suffix == '.json' else destination / 'result.json')
    if not result_path.is_relative_to(base) or result_path.exists():
        raise ValueError('Output must be a fresh JSON inside ' + str(base))
    with ExitStack() as stack:
        for path in [ROOT / 'artifacts/runs/kobayashi-jupyter-20261119/.gpu-series.lock', base / '.benchmark.lock']:
            path = project_path(path, 'GPU lock')
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open('a+'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('GPU/benchmark lock busy: ' + str(path)) from None
        source = project_path(ROOT / 'configs/experiments/garg.yaml', must_exist=True)
        config = configuration.load(source, seed=1)
        train = config['train']
        if (config['experiment'] != 'garg' or train['batch_size'] != 64 or train['precision'] != 'fp32'
                or not train['deterministic'] or train['threads'] != 4 or train['grad_clip'] != 0
                or train['optimizer'] != 'adam' or train['scheduler'] != 'constant' or train['warmup_steps']):
            raise ValueError('Expected unchanged full Garg configuration: batch64, FP32, deterministic, Adam, clip inf, constant LR')
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = {'purpose': 'technical_timing_only; not a scientific experiment', 'state': 'running',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'config_resolved': config,
                  'config_hash': configuration.digest(config), 'code_hash': code_hash(),
                  'script_sha256': sha(project_path(__file__)), 'sample_step': args.sample_step,
                  'warmup_updates': args.warmup, 'timed_updates': args.timed, 'repeats': args.repeats,
                  'cases': [], 'limitations': [
                      'Precomputed excludes its measured generation cost: upper bound, not full-pipeline speedup.',
                      'Short counterbalanced cases do not establish long-run thermal throughput.',
                      'IORegistry polling adds equal monitoring load; process time includes Python producer/monitor threads, not the ioreg child.',
                      'Losses may differ under MPS nondeterminism despite identical weights and input hashes.',
                      'No validation, checkpoints, scientific outputs, model changes or gradient shortcuts.']}
        write_json(result_path, result)
        try:
            device = engine.setup(config, 'mps')
            result['environment'] = environment(device)
            for repeat in range(args.repeats):
                for mode in args.modes if repeat % 2 == 0 else reversed(args.modes):
                    print(f'timing repeat {repeat + 1}/{args.repeats}: {mode}', flush=True)
                    case = run_case(config, args, mode)
                    result['cases'].append({**case, 'repeat': repeat + 1, 'order': len(result['cases']) + 1})
                    write_json(result_path, result)
                    gc.collect()
                    torch.mps.empty_cache()
            print('CPU sample and intrusive component profiles', flush=True)
            result['profiles'] = profiles(config, args)
            for key in ['initial_weights_sha256', 'first_timed_batch_sha256', 'last_timed_batch_sha256', 'generator_after_sha256']:
                if len({case[key] for case in result['cases']}) != 1:
                    raise RuntimeError('Benchmark cases differ: ' + key)
            result['state'] = 'completed_timing_only'
        except BaseException as error:
            result.update(state='failed', error=repr(error))
            raise
        finally:
            result['finished_at'] = datetime.now(timezone.utc).isoformat()
            write_json(result_path, result)
            gc.collect()
        print(result_path, flush=True)


if __name__ == '__main__':
    main()
