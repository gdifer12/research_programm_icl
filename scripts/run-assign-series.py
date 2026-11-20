#!/usr/bin/env python3
"""Sequential Assign and Add runs launched by the editable Jupyter notebook."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

from icl_lab import config as configuration
from icl_lab.runtime import ROOT, project_path, require_local_environment
from icl_lab.storage import code_hash, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    path = project_path(path, 'series JSON input')
    return json.loads(path.read_text()) if path.is_file() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    require_local_environment()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Seeds must be distinct non-negative integers')
    root = project_path(args.run_root, 'series directory')
    root.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'configs/experiments/assign_add.yaml'
    configs = {seed: configuration.load(config_path, seed=seed) for seed in args.seeds}
    identity = {'schema_version': 1, 'experiment': 'assign_add', 'seeds': args.seeds,
                'device': 'mps', 'code_hash': code_hash(),
                'config_hashes': {str(s): configuration.digest(c) for s, c in configs.items()}}
    child = None
    interrupted = 0

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with open(project_path(root / '.writer.lock', 'series lock'), 'a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This series already has a live worker or child process') from None
        manifest_path = root / 'series-manifest.json'
        if manifest_path.exists():
            saved = read_json(manifest_path)
            if any(saved.get(k) != value for k, value in identity.items()):
                raise RuntimeError('Series configuration/code changed: use a new run directory')
        else:
            write_json(manifest_path, {**identity, 'created_at': now()})
        state_path = root / 'series-state.json'
        previous = read_json(state_path)
        state = {**identity, 'state': 'running', 'pid': os.getpid(),
                 'started_at': previous.get('started_at', now()), 'worker_started_at': now(),
                 'stages': previous.get('stages', [])}

        def save(**updates):
            state.update(updates, updated_at=now())
            write_json(state_path, state)

        def execute(seed, stage, command):
            nonlocal child
            if interrupted:
                raise InterruptedError('Series interrupted before next stage')
            log_path = project_path(root / f'seed-{seed}-{stage}.log', 'stage log')
            entry = {'seed': seed, 'stage': stage, 'command': command,
                     'started_at': now(), 'log': str(log_path.relative_to(ROOT))}
            state['stages'].append(entry)
            save(seed=seed, stage=stage)
            started = time.monotonic()
            with log_path.open('a') as log:
                # The child keeps the series lock if the worker itself is killed.
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(stream.fileno(),))
                if interrupted:
                    stop(interrupted, None)
                save(child_pid=child.pid)
                returncode = child.wait()
            child = None
            entry.update(finished_at=now(), elapsed_seconds=time.monotonic() - started,
                         returncode=returncode)
            save(child_pid=None)
            if interrupted or returncode:
                raise RuntimeError(f'{stage} seed {seed} stopped (exit {returncode}); see {log_path}')

        save()
        try:
            for seed, config in configs.items():
                run = project_path(root / f'seed-{seed}', 'seed directory')
                saved_config = project_path(run / 'config.yaml', 'saved configuration')
                status = read_json(run / 'status.json')
                prefix = [str(ROOT / 'scripts/local-python'), '-m', 'icl_lab']
                if saved_config.exists():
                    if configuration.digest(yaml.safe_load(saved_config.read_text())) != configuration.digest(config):
                        raise RuntimeError(f'Saved configuration differs for seed {seed}')
                    manifest = read_json(run / 'manifest.json')
                    if manifest.get('code_hash') != identity['code_hash']:
                        raise RuntimeError(f'Saved code differs for seed {seed}')
                completed = status.get('state') == 'completed' and status.get('step') == config['train']['steps']
                if not completed:
                    command = prefix + ['train', '--run-dir', str(run), '--device', 'mps']
                    if saved_config.exists():
                        command += ['--resume', 'last']
                    else:
                        command += ['--config', str(config_path), '--seed', str(seed)]
                    execute(seed, 'train', command)
                    status = read_json(run / 'status.json')
                if status.get('state') != 'completed' or status.get('step') != config['train']['steps']:
                    raise RuntimeError(f'Seed {seed} is paused or incomplete; refusing to evaluate as final')
                for split in ['id_test', 'test']:
                    # The existing evaluator validates checkpoint/bank/code/device and reuses its cache.
                    execute(seed, f'evaluate-{split}', prefix + [
                        'evaluate', '--run-dir', str(run), '--checkpoint', 'all',
                        '--split', split, '--device', 'mps'])
                    summaries = list(project_path(run / 'evaluations', 'evaluations directory').glob(f'{split}-*/summary.json'))
                    if not any(read_json(p).get('step') == config['train']['steps'] for p in summaries):
                        raise RuntimeError(f'Missing full-budget evaluation for seed {seed}, {split}')
                    if not summaries or not all(project_path(p.with_name('predictions.npz'), 'predictions').is_file() for p in summaries):
                        raise RuntimeError(f'Missing evaluation artifacts for seed {seed}, {split}')
            save(state='completed', seed=None, stage='ready-for-notebook-analysis', finished_at=now())
        except BaseException as error:
            save(state='interrupted' if interrupted else 'failed', error=str(error), finished_at=now())
            raise


if __name__ == '__main__':
    main()
