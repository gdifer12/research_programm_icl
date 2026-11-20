"""One explicitly authorized Garg continuation: preserved 200000 -> completed 500001, one seed."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import shutil
import tempfile
import time

from icl_lab.runtime import ROOT, project_path, require_local_environment

spec = importlib.util.spec_from_file_location('garg_pilot_source', project_path(ROOT / 'scripts/garg_pilot.py', must_exist=True))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
io = pilot.io
RECORD = ROOT / 'artifacts/records/garg-200k-20261119.json'
START, TARGET = 200000, 500001
STEPS = (300000, 400000, TARGET)


def capture_source_record(directory):
    """Preserve exact record bytes once; later lifecycle edits cannot rewrite this input."""
    snapshot = project_path(directory / 'source-record.json', 'source record snapshot')
    if snapshot.exists():
        return snapshot
    data = project_path(RECORD, must_exist=True).read_bytes()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.source-record-', delete=False) as stream:
            temporary = project_path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, snapshot)  # Atomic creation, no overwrite of an earlier snapshot.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return snapshot


def source_identity(run):
    snapshot = project_path(stage_root(run) / 'source-record.json')
    record_path = snapshot if snapshot.exists() else project_path(RECORD, must_exist=True)
    record_bytes = record_path.read_bytes()
    record = json.loads(record_bytes)
    original = pilot.identity()
    saved = pilot.check_manifest(run.parent, original)
    if not saved or not record or project_path(ROOT / record['run_dir']) != run:
        raise RuntimeError('Missing/mismatched source 200k record')
    authorization = record.get('authorization', {})
    if (record.get('phase') != 'analysis_completed' or authorization.get('seed') != 1
            or authorization.get('target_step') != START
            or record.get('full_config_steps') != 500001 or record.get('device') != 'mps'
            or record.get('precision') != 'fp32' or record.get('technical_smoke') is not False):
        raise RuntimeError('Source 200k record has unexpected settings')
    manifest = io.read_json(run / 'manifest.json')
    for key in ['code_hash', 'config_hash']:
        if not record.get(key) or record[key] != saved.get(key) or record[key] != manifest.get(key):
            raise RuntimeError('Original source/config identity differs: ' + key)
    if saved.get('saved_config_sha256') != io.sha(run / 'config.yaml'):
        raise RuntimeError('Original saved configuration changed')
    checkpoint = project_path(run / 'checkpoints/pilot-00200000.pt', must_exist=True)
    preserved = record.get('preserved_checkpoint', {})
    if (not preserved.get('path') or project_path(ROOT / preserved['path']) != checkpoint
            or preserved.get('step') != START or preserved.get('full_resumable') is not True
            or preserved.get('model_optimizer_rng_present') is not True
            or io.sha(checkpoint) != preserved.get('sha256')
            or project_path(checkpoint.with_suffix('.pt.sha256'), must_exist=True).read_text().strip() != preserved['sha256']):
        raise RuntimeError('Preserved 200000-step checkpoint differs from the source record')
    bank_path = project_path(run / 'banks/manifest.json', must_exist=True)
    bank_digest = io.sha(bank_path)
    if record.get('analysis_artifact_sha256', {}).get(str(bank_path.relative_to(ROOT))) != bank_digest:
        raise RuntimeError('Bank manifest differs from the source 200k record')
    banks = io.read_json(bank_path)
    if banks.get('config_hash') != record['config_hash'] or not banks.get('banks'):
        raise RuntimeError('Bank configuration differs from the source record')
    bank_hashes = {}
    for split, bank in banks['banks'].items():
        if not bank.get('file') or not bank.get('sha256') or io.sha(project_path(run / 'banks' / bank['file'])) != bank['sha256']:
            raise RuntimeError('Bank file differs: ' + split)
        bank_hashes[split] = bank['sha256']
    evidence = record.get('source_checks', {}).get(str(START), {})
    if (evidence.get('checkpoint_sha256') != preserved['sha256']
            or evidence.get('validation_bank_sha256') != bank_hashes.get('validation')):
        raise RuntimeError('Source checkpoint/validation evidence differs')
    return {'schema_version': 1, 'experiment': 'garg', 'seed': 1, 'device': 'mps',
            'start_step': START, 'target_step': TARGET, 'steps': list(STEPS), 'additional_updates': TARGET - START,
            'full_training_steps': 500001, 'code_hash': record['code_hash'], 'config_hash': record['config_hash'],
            'config_source_sha256': original['config_source_sha256'],
            'saved_config_sha256': saved['saved_config_sha256'], 'source_checkpoint': str(checkpoint),
            'source_checkpoint_sha256': preserved['sha256'], 'source_record': str(snapshot),
            'source_record_live': str(project_path(RECORD)),
            'source_record_sha256': hashlib.sha256(record_bytes).hexdigest(),
            'bank_manifest_sha256': bank_digest, 'bank_sha256': bank_hashes,
            'worker_script_sha256': io.sha(__file__)}


def stage_root(run):
    return project_path(run.parent / 'stages/500k', 'continuation stage')


def check_stage(directory, expected):
    saved = io.read_json(directory / 'stage-manifest.json')
    if saved and any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Continuation identity changed; inspect before continuing')
    return saved


def boundary_state(step):
    return 'completed' if step == TARGET else 'paused'


def at_boundary(run, step):
    status = io.read_json(run / 'status.json')
    return status.get('step') == step and status.get('state') == boundary_state(step)


def checkpoint_path(run, step):
    return project_path(run / f'checkpoints/pilot-{step:08d}.pt')


def validation_summary(run, expected, step):
    """Match one immutable boundary independently of later changes to last.pt."""
    checkpoint = checkpoint_path(run, step)
    checksum = project_path(checkpoint.with_suffix('.pt.sha256'))
    if not checkpoint.is_file() or not checksum.is_file():
        return None
    digest = io.sha(checkpoint)
    if digest != checksum.read_text().strip():
        return None
    bank = io.read_json(run / 'banks/manifest.json').get('banks', {}).get('validation', {})
    directory = project_path(run / 'evaluations')
    if not directory.is_dir() or not bank.get('file') or not bank.get('sha256'):
        return None
    bank_path = project_path(run / 'banks' / bank['file'])
    if not bank_path.is_file() or io.sha(bank_path) != bank['sha256']:
        return None
    for candidate in directory.iterdir():
        if not candidate.name.startswith('validation-'):
            continue
        entry = project_path(candidate, 'validation directory')
        if not entry.is_dir():
            continue
        path = project_path(entry / 'summary.json')
        value = io.read_json(path)
        if (value.get('step') == step and value.get('split') == 'validation'
                and value.get('checkpoint_sha256') == digest and value.get('bank_sha256') == bank['sha256']
                and value.get('code_hash') == expected['code_hash'] and value.get('predictions')
                and project_path(path.with_name('predictions.npz')).is_file()):
            return str(path)
    return None


def completed_snapshots(run, expected, state):
    """Idempotence requires all three preserved and evaluated boundaries."""
    snapshots = state.get('snapshots', [])
    if (state.get('state') != 'stage_completed' or state.get('step') != TARGET
            or [item.get('step') for item in snapshots] != list(STEPS) or not at_boundary(run, TARGET)):
        return False
    for snapshot in snapshots:
        step = snapshot['step']
        path = checkpoint_path(run, step)
        summary = validation_summary(run, expected, step)
        if (not summary or snapshot.get('validation_summary') != summary
                or snapshot.get('checkpoint') != str(path)
                or snapshot.get('checkpoint_sha256') != io.sha(path)
                or snapshot.get('engine_state') != boundary_state(step)):
            return False
    return io.sha(project_path(run / 'checkpoints/last.pt', must_exist=True)) == snapshots[-1]['checkpoint_sha256']


def launch_continuation(run_dir=None):
    """Detach the authorized queue once; partial/failed queues require inspection."""
    require_local_environment()
    run = pilot.resolve(run_dir)
    directory = stage_root(run)
    directory.mkdir(parents=True, exist_ok=True)
    info = {'run_dir': str(run), 'stage_root': str(directory), 'log': str(project_path(directory / 'worker.log'))}
    with ExitStack() as stack:
        try:
            writer = stack.enter_context(pilot.acquire(run.parent / '.writer.lock'))
        except BlockingIOError:
            return {**io.read_json(directory / 'stage-state.json'), **info, 'state': 'running', 'already_running': True}
        expected = source_identity(run)
        check_stage(directory, expected)
        previous = io.read_json(directory / 'stage-state.json')
        if completed_snapshots(run, expected, previous):
            return {**previous, **info, 'already_completed': True}
        if previous or not at_boundary(run, START):
            return {**previous, **info, 'state': 'incomplete', 'requires_inspection': True,
                    'reason': 'Require untouched paused 200000 source; automatic redrive is disabled'}
        try:
            gpu = stack.enter_context(pilot.acquire(pilot.GPU_LOCK))
        except BlockingIOError:
            return {**info, 'state': 'blocked', 'pid': None, 'reason': 'Another GPU series or benchmark is running'}
        capture_source_record(directory)
        expected = source_identity(run)
        check_stage(directory, expected)
        pending = {**expected, 'state': 'starting', 'pid': None, 'stage': 'launch', 'step': START,
                   'engine_state': 'paused', 'started_at': io.now(), 'stages': [], 'snapshots': []}
        io.write_json(directory / 'stage-state.json', pending)
        command = [str(project_path(ROOT / 'scripts/local-python', must_exist=True)), str(project_path(__file__)),
                   '--run-dir', str(run), '--lock-fd', str(writer.fileno()), '--gpu-lock-fd', str(gpu.fileno())]
        try:
            with project_path(info['log']).open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(writer.fileno(), gpu.fileno()))
        except BaseException as error:
            io.write_json(directory / 'stage-state.json', {**pending, 'state': 'failed', 'error': repr(error)})
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
    from icl_lab.storage import atomic_bytes, code_hash, load_torch
    run = pilot.resolve(args.run_dir)
    directory = stage_root(run)
    directory.mkdir(parents=True, exist_ok=True)
    child, interrupted = None, 0
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
            writer = stack.enter_context(pilot.acquire(run.parent / '.writer.lock', args.lock_fd))
            gpu = stack.enter_context(pilot.acquire(pilot.GPU_LOCK, args.gpu_lock_fd))
            for number in handlers:
                signal.signal(number, stop)
            previous = io.read_json(directory / 'stage-state.json')
            state = {'state': 'running', 'pid': os.getpid(), 'child_pid': None, 'step': START,
                     'started_at': previous.get('started_at', io.now()),
                     'stages': previous.get('stages', []), 'snapshots': previous.get('snapshots', [])}

            def save(**updates):
                state.update(updates, updated_at=io.now())
                io.write_json(directory / 'stage-state.json', state)

            def execute(stage, boundary, command):
                nonlocal child
                if interrupted:
                    raise InterruptedError('Continuation interrupted before next stage')
                log = project_path(directory / f'{stage}-{boundary:08d}.log')
                entry = {'stage': stage, 'boundary_step': boundary, 'command': command,
                         'log': str(log), 'started_at': io.now()}
                state['stages'].append(entry)
                save(stage=stage, boundary_step=boundary)
                started = time.monotonic()
                with log.open('a') as output:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                             stderr=subprocess.STDOUT, start_new_session=True,
                                             pass_fds=(writer.fileno(), gpu.fileno()))
                    if interrupted:
                        stop(interrupted, None)
                    save(child_pid=child.pid)
                    result = child.wait()
                child = None
                entry.update(returncode=result, elapsed_seconds=time.monotonic() - started, finished_at=io.now())
                save(child_pid=None)
                if interrupted or result:
                    raise RuntimeError(f'{stage} stopped (exit {result}); see {log}')

            try:
                capture_source_record(directory)
                expected = source_identity(run)
                if previous.get('source_record_sha256', expected['source_record_sha256']) != expected['source_record_sha256']:
                    raise RuntimeError('Source record snapshot changed since launch')
                config = configuration.load(pilot.CONFIG, seed=1)
                if (configuration.digest(config) != expected['config_hash'] or code_hash() != expected['code_hash']
                        or config['train']['steps'] != TARGET or config['train']['precision'] != 'fp32'
                        or config['model_seed'] != 1 or config['train_seed'] != 10001 or config.get('technical_smoke')
                        or configuration.digest(yaml.safe_load(project_path(run / 'config.yaml').read_text())) != expected['config_hash']):
                    raise RuntimeError('Saved/source configuration or code differs from the original pilot')
                manifest = check_stage(directory, expected)
                if completed_snapshots(run, expected, previous):
                    return
                launched = args.lock_fd is not None and previous.get('state') == 'starting' and not previous.get('stages')
                if not at_boundary(run, START) or (previous and not launched):
                    raise RuntimeError('Continuation requires inspection; automatic redrive is disabled')
                state.update(expected, engine_state='paused')
                manifest.update(expected)
                io.write_json(directory / 'stage-manifest.json', manifest)

                def checked_payload(path, step):
                    payload = load_torch(path, full=True)  # Trusted project-local file, CPU only.
                    if (payload.get('step') != step or not payload.get('resumable')
                            or payload.get('code_hash') != expected['code_hash']
                            or payload.get('config_hash') != expected['config_hash']
                            or payload.get('training_device') != 'mps'
                            or not {'optimizer', 'rng', 'scaler'}.issubset(payload)):
                        raise RuntimeError('Checkpoint is not the expected full MPS resume state')

                prefix = [str(ROOT / 'scripts/local-python'), '-m', 'icl_lab']
                last = project_path(run / 'checkpoints/last.pt', must_exist=True)
                resume = project_path(expected['source_checkpoint'], must_exist=True)
                resume_digest = expected['source_checkpoint_sha256']
                start = START
                for boundary in STEPS:
                    if not at_boundary(run, start) or io.sha(last) != resume_digest or io.sha(resume) != resume_digest:
                        raise RuntimeError('Current state differs from preserved source at chunk boundary')
                    checked_payload(resume, start)
                    execute('train', boundary, prefix + ['train', '--run-dir', str(run), '--device', 'mps',
                                                        '--resume', str(resume), '--stop-after', str(boundary - start)])
                    if not at_boundary(run, boundary):
                        raise RuntimeError(f'Continuation did not reach exact {boundary_state(boundary)} boundary {boundary}')
                    checked_payload(last, boundary)
                    final = checkpoint_path(run, boundary)
                    digest = io.sha(last)
                    if final.exists() and io.sha(final) != digest:
                        raise RuntimeError(f'Existing preserved {boundary}-step checkpoint differs')
                    if not final.exists():
                        os.link(last, final)  # Engine atomically rotates last; preserve this full state.
                    checksum = project_path(final.with_suffix('.pt.sha256'))
                    if checksum.exists() and checksum.read_text().strip() != digest:
                        raise RuntimeError('Existing preserved checkpoint checksum differs')
                    if not checksum.exists():
                        atomic_bytes(checksum, (digest + '\n').encode())
                    save(step=boundary, engine_state=boundary_state(boundary), checkpoint=str(final), checkpoint_sha256=digest)
                    execute('evaluate-validation', boundary, prefix + ['evaluate', '--run-dir', str(run), '--device', 'mps',
                                                                     '--checkpoint', str(final), '--split', 'validation'])
                    summary = validation_summary(run, expected, boundary)
                    if not summary:
                        raise RuntimeError(f'Missing matching {boundary}-step validation artifacts')
                    state['snapshots'].append({'step': boundary, 'engine_state': boundary_state(boundary),
                                               'checkpoint': str(final), 'checkpoint_sha256': digest,
                                               'validation_summary': summary})
                    save(validation_summary=summary)
                    start, resume, resume_digest = boundary, final, digest
                save(state='stage_completed', step=TARGET, engine_state='completed',
                     stage='ready-for-notebook-analysis', finished_at=io.now())
            except BaseException as error:
                save(state='interrupted' if interrupted else 'failed', error=repr(error), finished_at=io.now())
                raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()
