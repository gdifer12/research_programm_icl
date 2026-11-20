"""Launch one 50000-step Garg pilot; notebook imports use only the standard library."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import shutil
import time

from icl_lab.runtime import ROOT, project_path, require_local_environment

spec = importlib.util.spec_from_file_location('pilot_io', project_path(ROOT / 'scripts/notebook_series.py', must_exist=True))
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
DEFAULT_RUN = ROOT / 'artifacts/runs/garg-pilot-20261119/seed-1'
GPU_LOCK = ROOT / 'artifacts/runs/kobayashi-jupyter-20261119/.gpu-series.lock'
CONFIG = ROOT / 'configs/experiments/garg.yaml'
STOP = 50000


def resolve(run_dir=None):
    path = Path(run_dir) if run_dir is not None else DEFAULT_RUN
    path = project_path(path if path.is_absolute() else ROOT / path, 'pilot run')
    if path.name != 'seed-1':
        raise ValueError('Pilot run directory must end with seed-1')
    return path


def identity():
    files = []
    for candidate in project_path(ROOT / 'src', must_exist=True).rglob('*'):
        if '__pycache__' in candidate.parts:
            continue
        path = project_path(candidate, 'source file')
        if path.is_file():
            files.append(path)
    code = hashlib.sha256(''.join(str(p.relative_to(ROOT)) + io.sha(p) for p in sorted(files)).encode()).hexdigest()
    return {'schema_version': 1, 'experiment': 'garg', 'seed': 1, 'device': 'mps',
            'stop_after': STOP, 'full_training_steps': 500001, 'code_hash': code,
            'config_source_sha256': io.sha(CONFIG), 'worker_script_sha256': io.sha(__file__)}


def check_manifest(root, expected):
    saved = io.read_json(root / 'pilot-manifest.json')
    if saved and any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Pilot code/config changed; inspect the existing run before continuing')
    return saved


def phase(run):
    status = io.read_json(run / 'status.json')
    config = project_path(run / 'config.yaml')
    if status.get('state') == 'paused' and status.get('step') == STOP and config.is_file():
        return 'evaluate'
    if status or (run.is_dir() and any(run.iterdir())):
        return 'incomplete'
    return 'fresh'


def validation_summary(run, expected):
    checkpoint = project_path(run / 'checkpoints/last.pt')
    directory = project_path(run / 'evaluations')
    bank = io.read_json(run / 'banks/manifest.json').get('banks', {}).get('validation', {})
    if not checkpoint.is_file() or not directory.is_dir() or not bank.get('file') or not bank.get('sha256'):
        return None
    bank_path = project_path(run / 'banks' / bank['file'])
    if not bank_path.is_file() or io.sha(bank_path) != bank['sha256']:
        return None
    checkpoint_hash = io.sha(checkpoint)
    for candidate in directory.iterdir():
        if not candidate.name.startswith('validation-'):
            continue
        entry = project_path(candidate, 'validation output')
        if not entry.is_dir():
            continue
        path = project_path(entry / 'summary.json')
        value = io.read_json(path)
        if (value.get('step') == STOP and value.get('split') == 'validation'
                and value.get('code_hash') == expected['code_hash']
                and value.get('checkpoint_sha256') == checkpoint_hash and value.get('bank_sha256') == bank['sha256']
                and value.get('predictions') and project_path(path.with_name('predictions.npz')).is_file()):
            return str(path)
    return None


def acquire(path, descriptor=None):
    path = project_path(path, 'pilot lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    if descriptor is not None:
        actual, expected = os.fstat(descriptor), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('Inherited descriptor does not match the pilot lock')
    stream = path.open('a+') if descriptor is None else os.fdopen(descriptor, 'a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except BaseException:
        stream.close()
        raise


def launch_pilot(run_dir=None):
    require_local_environment()
    run = resolve(run_dir)
    root = run.parent
    root.mkdir(parents=True, exist_ok=True)
    expected = identity()
    saved = check_manifest(root, expected)
    previous = io.read_json(root / 'pilot-state.json')
    info = {'run_dir': str(run), 'log': str(project_path(root / 'worker.log'))}
    with ExitStack() as stack:
        try:
            writer = stack.enter_context(acquire(root / '.writer.lock'))
        except BlockingIOError:
            return {**previous, **info, 'state': 'running', 'already_running': True}
        current = phase(run)
        if current == 'incomplete' or (current == 'fresh' and previous):
            return {**previous, **info, 'state': 'incomplete', 'requires_inspection': True,
                    'reason': 'Existing pilot is not paused exactly at 50000; automatic restart/resume is disabled'}
        if previous.get('state') == 'pilot_completed' and current == 'evaluate':
            if saved.get('saved_config_sha256') != io.sha(run / 'config.yaml'):
                raise RuntimeError('Saved pilot configuration changed')
            summary = validation_summary(run, expected)
            if summary:
                return {**previous, **info, 'already_completed': True, 'validation_summary': summary}
        try:
            gpu = stack.enter_context(acquire(GPU_LOCK))
        except BlockingIOError:
            return {**info, 'state': 'blocked', 'pid': None, 'reason': 'Another GPU series or benchmark is running'}
        pending = {**expected, 'state': 'starting', 'pid': None, 'stage': 'launch',
                   'started_at': previous.get('started_at', io.now()), 'stages': previous.get('stages', [])}
        io.write_json(root / 'pilot-state.json', pending)
        command = [str(project_path(ROOT / 'scripts/local-python', must_exist=True)), str(project_path(__file__)),
                   '--run-dir', str(run), '--lock-fd', str(writer.fileno()), '--gpu-lock-fd', str(gpu.fileno())]
        try:
            with project_path(info['log']).open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(writer.fileno(), gpu.fileno()))
        except BaseException as error:
            io.write_json(root / 'pilot-state.json', {**pending, 'state': 'failed', 'error': repr(error)})
            raise
        warning = None
        try:
            if caffeinate := shutil.which('caffeinate'):
                subprocess.Popen([caffeinate, '-i', '-w', str(child.pid)], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            warning = str(error)
        return {**info, 'state': 'running', 'pid': child.pid, 'warning': warning}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--gpu-lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_local_environment()
    if (args.lock_fd is None) != (args.gpu_lock_fd is None):
        parser.error('Both inherited descriptors are required together')
    import yaml
    from icl_lab import config as configuration
    from icl_lab.storage import code_hash
    run = resolve(args.run_dir)
    root, child, interrupted = run.parent, None, 0
    handlers = {number: signal.getsignal(number) for number in [signal.SIGTERM, signal.SIGINT]}

    def stop(number, _frame):
        nonlocal interrupted
        interrupted = number
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, number)
            except ProcessLookupError:
                pass

    try:
        with ExitStack() as stack:
            writer = stack.enter_context(acquire(root / '.writer.lock', args.lock_fd))
            gpu = stack.enter_context(acquire(GPU_LOCK, args.gpu_lock_fd))
            for number in handlers:
                signal.signal(number, stop)
            previous = io.read_json(root / 'pilot-state.json')
            state = {'state': 'running', 'pid': os.getpid(), 'child_pid': None,
                     'started_at': previous.get('started_at', io.now()), 'stages': previous.get('stages', [])}

            def save(**updates):
                state.update(updates, updated_at=io.now())
                io.write_json(root / 'pilot-state.json', state)

            def execute(stage, command):
                nonlocal child
                if interrupted:
                    raise InterruptedError('Pilot interrupted before the next stage')
                log = project_path(root / f'{stage}.log')
                entry = {'stage': stage, 'command': command, 'log': str(log), 'started_at': io.now()}
                state['stages'].append(entry)
                save(stage=stage)
                started = time.monotonic()
                with log.open('a') as output:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                             stderr=subprocess.STDOUT, start_new_session=True,
                                             pass_fds=(writer.fileno(), gpu.fileno()))
                    if interrupted:
                        stop(interrupted, None)
                    save(child_pid=child.pid)
                    code = child.wait()
                child = None
                entry.update(returncode=code, elapsed_seconds=time.monotonic() - started, finished_at=io.now())
                save(child_pid=None)
                if interrupted or code:
                    raise RuntimeError(f'{stage} stopped (exit {code}); see {log}')

            try:
                expected = identity()
                if expected['code_hash'] != code_hash():
                    raise RuntimeError('Launcher/training source digest mismatch')
                config = configuration.load(CONFIG, seed=1)
                if (config['experiment'] != 'garg' or config['train']['steps'] != 500001
                        or config['train']['precision'] != 'fp32' or config['train']['batch_size'] != 64
                        or config.get('technical_smoke')):
                    raise ValueError('Expected unchanged full Garg configuration')
                manifest = check_manifest(root, expected)
                if manifest.get('config_hash', configuration.digest(config)) != configuration.digest(config):
                    raise RuntimeError('Pilot resolved configuration changed')
                manifest.update(expected, config_hash=configuration.digest(config))
                state.update(expected)
                current = phase(run)
                if current == 'incomplete' or (current == 'fresh' and previous and args.lock_fd is None):
                    raise RuntimeError('Existing pilot requires inspection; automatic restart/resume is disabled')
                io.write_json(root / 'pilot-manifest.json', manifest)
                prefix = [str(ROOT / 'scripts/local-python'), '-m', 'icl_lab']
                if current == 'fresh':
                    execute('train', prefix + ['train', '--config', str(CONFIG), '--run-dir', str(run),
                                              '--seed', '1', '--device', 'mps', '--stop-after', str(STOP)])
                if phase(run) != 'evaluate':
                    raise RuntimeError('Pilot did not pause exactly at 50000; refusing final evaluation')
                saved_config = project_path(run / 'config.yaml', must_exist=True)
                if (configuration.digest(yaml.safe_load(saved_config.read_text())) != configuration.digest(config)
                        or io.read_json(run / 'manifest.json').get('code_hash') != expected['code_hash']):
                    raise RuntimeError('Saved run configuration/code differs')
                manifest['saved_config_sha256'] = io.sha(saved_config)
                io.write_json(root / 'pilot-manifest.json', manifest)
                execute('evaluate-validation', prefix + ['evaluate', '--run-dir', str(run), '--checkpoint', 'last',
                                                        '--split', 'validation', '--device', 'mps'])
                summary = validation_summary(run, expected)
                if not summary:
                    raise RuntimeError('Missing matching 50000-step validation artifacts')
                save(state='pilot_completed', step=STOP, stage='ready-for-notebook-analysis',
                     validation_summary=summary, finished_at=io.now())
            except BaseException as error:
                save(state='interrupted' if interrupted else 'failed', error=repr(error), finished_at=io.now())
                raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()
