#!/usr/bin/env python3
"""Sequential full Kobayashi runs, launched by notebook_series.launch_series."""
import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

from icl_lab import config as configuration
from icl_lab.runtime import ROOT, project_path, require_local_environment
from icl_lab.storage import code_hash

spec = importlib.util.spec_from_file_location('notebook_series', project_path(ROOT / 'scripts/notebook_series.py', must_exist=True))
series = importlib.util.module_from_spec(spec)
spec.loader.exec_module(series)


def acquire(path, descriptor=None):
    path = project_path(path, 'series lock')
    if descriptor is None:
        stream = path.open('a+')
    else:
        actual, expected = os.fstat(descriptor), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('Inherited descriptor does not match the series lock')
        stream = os.fdopen(descriptor, 'a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=series.CONFIGS, required=True)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--gpu-lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_local_environment()
    if (args.lock_fd is None) != (args.gpu_lock_fd is None):
        parser.error('Both inherited lock descriptors are required together')
    root = series.resolve_root(args.variant, args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    child = None
    interrupted = 0
    handlers = {number: signal.getsignal(number) for number in [signal.SIGTERM, signal.SIGINT]}

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for number in handlers:
            signal.signal(number, stop)
        with ExitStack() as stack:
            try:
                writer = stack.enter_context(acquire(root / '.writer.lock', args.lock_fd))
                gpu = stack.enter_context(acquire(root.parent / '.gpu-series.lock', args.gpu_lock_fd))
            except BlockingIOError:
                raise RuntimeError('A Kobayashi worker or child already holds the GPU/series lock') from None
            previous = series.read_json(root / 'series-state.json')
            state = {'variant': args.variant, 'seeds': args.seeds, 'state': 'running',
                     'pid': os.getpid(), 'child_pid': None,
                     'started_at': previous.get('started_at', series.now()), 'worker_started_at': series.now(),
                     'stages': previous.get('stages', [])}

            def save(**updates):
                state.update(updates, updated_at=series.now())
                series.write_json(root / 'series-state.json', state)

            def execute(seed, stage, command):
                nonlocal child
                if interrupted:
                    raise InterruptedError('Series interrupted before the next stage')
                log_path = project_path(root / f'seed-{seed}-{stage}.log', 'stage log')
                entry = {'seed': seed, 'stage': stage, 'command': command, 'started_at': series.now(),
                         'log': str(log_path.relative_to(ROOT))}
                state['stages'].append(entry)
                save(seed=seed, stage=stage)
                started = time.monotonic()
                with log_path.open('a') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                             pass_fds=(writer.fileno(), gpu.fileno()))
                    if interrupted:
                        stop(interrupted, None)
                    save(child_pid=child.pid)
                    returncode = child.wait()
                child = None
                entry.update(finished_at=series.now(), elapsed_seconds=time.monotonic() - started,
                             returncode=returncode)
                save(child_pid=None)
                if interrupted or returncode:
                    raise RuntimeError(f'{stage} seed {seed} stopped (exit {returncode}); see {log_path}')

            try:
                expected = series.identity(args.variant, args.seeds)
                if expected['code_hash'] != code_hash():
                    raise RuntimeError('Launcher and training source digests differ')
                config_path = project_path(ROOT / 'configs/experiments' / series.CONFIGS[args.variant], must_exist=True)
                configs = {seed: configuration.load(config_path, seed=seed) for seed in args.seeds}
                for config in configs.values():
                    if (config['experiment'] != 'kobayashi' or config['model']['variant'] != args.variant
                            or config['train']['steps'] != series.STEPS or config['train']['batch_size'] != 128
                            or config['train']['precision'] != 'fp32' or config.get('technical_smoke')):
                        raise ValueError('Expected full Kobayashi variant: 100000 updates, batch 128, FP32')
                identity = {**expected, 'config_hashes': {str(seed): configuration.digest(config)
                                                       for seed, config in configs.items()}}
                state.update(identity)
                saved = series.check_identity(root, identity)
                series.check_dependency(root, expected)
                if not saved:
                    series.write_json(root / 'series-manifest.json', {**identity, 'created_at': series.now()})
                save()
                for seed, config in configs.items():
                    run = project_path(root / f'seed-{seed}', 'seed directory')
                    saved_config = project_path(run / 'config.yaml', 'saved configuration')
                    status = series.read_json(run / 'status.json')
                    prefix = [str(ROOT / 'scripts/local-python'), '-m', 'icl_lab']
                    if saved_config.exists():
                        if configuration.digest(yaml.safe_load(saved_config.read_text())) != configuration.digest(config):
                            raise RuntimeError(f'Saved configuration differs for seed {seed}')
                        if series.read_json(run / 'manifest.json').get('code_hash') != identity['code_hash']:
                            raise RuntimeError(f'Saved code differs for seed {seed}')
                    elif status:
                        raise RuntimeError(f'Seed {seed} has status without a saved configuration')
                    done = status.get('state') == 'completed' and status.get('step') == series.STEPS
                    if not done:
                        command = prefix + ['train', '--run-dir', str(run), '--device', 'mps']
                        command += (['--resume', 'last'] if saved_config.exists()
                                    else ['--config', str(config_path), '--seed', str(seed)])
                        execute(seed, 'train', command)
                        status = series.read_json(run / 'status.json')
                    if status.get('state') != 'completed' or status.get('step') != series.STEPS:
                        raise RuntimeError(f'Seed {seed} is paused/incomplete; refusing final evaluation')
                    for split in ['id_test', 'test']:
                        execute(seed, f'evaluate-{split}', prefix + ['evaluate', '--run-dir', str(run),
                                '--checkpoint', 'all', '--split', split, '--device', 'mps'])
                    if not series.final_evaluations_exist(run, expected):
                        raise RuntimeError(f'Seed {seed}: missing full-budget ID/OOD evaluation artifacts')
                save(state='completed', seed=None, stage='ready-for-notebook-analysis', finished_at=series.now())
            except BaseException as error:
                save(state='interrupted' if interrupted else 'failed', error=str(error), finished_at=series.now())
                raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()
