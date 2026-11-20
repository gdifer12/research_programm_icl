import shutil
"""Verify the one authorized 100k -> 200k stage with CPU fixtures and mocked workers."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import pytest
import yaml

from icl_lab import config as configuration, storage

ROOT = Path(__file__).parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def continuation(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('garg_continue_200k', ROOT / 'scripts/garg_continue_200k.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.pilot.identity
    monkeypatch.setattr(module.pilot, 'identity', lambda: {**original(), 'code_hash': 'test-code'})
    monkeypatch.setattr(storage, 'code_hash', lambda: 'test-code')
    monkeypatch.setattr(module.pilot, 'GPU_LOCK', tmp_path / '.gpu.lock')
    monkeypatch.setattr(module, 'RECORD', tmp_path / 'record.json')
    return module


def payload(config, step):
    return {'step': step, 'resumable': True, 'training_device': 'mps', 'code_hash': 'test-code',
            'config_hash': configuration.digest(config), 'optimizer': {}, 'rng': {}, 'scaler': {}}


def source_run(module, run):
    run.mkdir(parents=True)
    config = configuration.load(module.pilot.CONFIG, seed=1)
    (run / 'config.yaml').write_text(yaml.safe_dump(config))
    write_json(run / 'manifest.json', {'code_hash': 'test-code', 'config_hash': configuration.digest(config)})
    write_json(run / 'status.json', {'state': 'paused', 'step': 100000})
    checkpoint = run / 'checkpoints/pilot-00100000.pt'
    storage.save_torch(checkpoint, payload(config, 100000))
    os.link(checkpoint, run / 'checkpoints/last.pt')
    (run / 'checkpoints/last.pt.sha256').write_text(module.io.sha(checkpoint))
    write_json(run.parent / 'pilot-manifest.json', {**module.pilot.identity(),
               'config_hash': configuration.digest(config), 'saved_config_sha256': module.io.sha(run / 'config.yaml')})
    bank = run / 'banks/validation.npz'
    bank.parent.mkdir()
    bank.write_bytes(b'bank fixture')
    bank_manifest = run / 'banks/manifest.json'
    write_json(bank_manifest, {'config_hash': configuration.digest(config),
                              'banks': {'validation': {'file': bank.name, 'sha256': module.io.sha(bank)}}})
    write_json(module.RECORD, {'run_dir': str(run), 'phase': 'analysis_completed',
               'authorization': {'seed': 1, 'target_step': 100000},
               'full_config_steps': 500001, 'device': 'mps', 'precision': 'fp32', 'technical_smoke': False,
               'code_hash': 'test-code', 'config_hash': configuration.digest(config),
               'preserved_checkpoint': {'path': str(checkpoint), 'sha256': module.io.sha(checkpoint),
                                       'step': 100000, 'resumable': True, 'optimizer_and_rng_present': True},
               'source_checks': {'100000': {'checkpoint_sha256': module.io.sha(checkpoint),
                                            'validation_bank_sha256': module.io.sha(bank)}},
               'analysis_artifact_sha256': {str(bank_manifest.relative_to(ROOT)): module.io.sha(bank_manifest)}})
    return config


def final_state(run, config, step=200000):
    storage.save_torch(run / 'checkpoints/last.pt', payload(config, step))
    write_json(run / 'status.json', {'state': 'paused', 'step': step})


def validation(module, run):
    directory = run / 'evaluations/validation-00200000-fixture'
    directory.mkdir(parents=True, exist_ok=True)
    prediction = directory / 'predictions.npz'
    prediction.write_bytes(b'prediction fixture')
    write_json(directory / 'summary.json', {'step': 200000, 'split': 'validation', 'code_hash': 'test-code',
               'checkpoint_sha256': module.io.sha(run / 'checkpoints/pilot-00200000.pt'),
               'bank_sha256': module.io.sha(run / 'banks/validation.npz'), 'predictions': str(prediction)})


@pytest.mark.parametrize('mode', ['start', 'target', 'below', 'above', 'changed-config', 'changed-code',
                                 'source-corrupted', 'bank-corrupted', 'last-advanced', 'child-failed',
                                 'missing-validation', 'partial-pair'])
def test_authorized_budget_and_failure_boundaries(continuation, tmp_path, monkeypatch, mode):
    module = continuation
    run = tmp_path / 'pilot/seed-1'
    config = source_run(module, run)
    monkeypatch.setattr(sys, 'argv', ['continue', '--run-dir', str(run)])
    if mode in ['target', 'missing-validation', 'partial-pair']:
        final_state(run, config)
    if mode in ['below', 'above']:
        final_state(run, config, 150000 if mode == 'below' else 200001)
    if mode == 'partial-pair':
        os.link(run / 'checkpoints/last.pt', run / 'checkpoints/pilot-00200000.pt')
    if mode == 'changed-config':
        changed = {**config, 'model_seed': 2}
        (run / 'config.yaml').write_text(yaml.safe_dump(changed))
    if mode == 'changed-code':
        monkeypatch.setattr(storage, 'code_hash', lambda: 'other-code')
    if mode == 'source-corrupted':
        (run / 'checkpoints/pilot-00100000.pt').write_bytes(b'corruption')
    if mode == 'bank-corrupted':
        (run / 'banks/validation.npz').write_bytes(b'corruption')
    if mode == 'last-advanced':
        storage.save_torch(run / 'checkpoints/last.pt', payload(config, 100001))
    calls = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            self.command = command
            calls.append(command)
            assert len(kwargs['pass_fds']) == 2 and kwargs['start_new_session']

        def wait(self):
            if mode == 'child-failed':
                return 7
            if 'train' in self.command:
                assert self.command[self.command.index('--stop-after') + 1] == '100000'
                assert Path(self.command[self.command.index('--resume') + 1]) == run / 'checkpoints/pilot-00100000.pt'
                assert not {'--config', '--seed', '--steps', '--allow-code-change', '--allow-device-change'} & set(self.command)
                final_state(run, config)
            elif mode != 'missing-validation':
                assert self.command[self.command.index('--split') + 1] == 'validation'
                assert Path(self.command[self.command.index('--checkpoint') + 1]) == run / 'checkpoints/pilot-00200000.pt'
                validation(module, run)
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    state_path = module.stage_root(run) / 'stage-state.json'
    if mode in ['start', 'target', 'partial-pair']:
        module.main()
        state = json.loads(state_path.read_text())
        assert state['state'] == 'stage_completed' and state['step'] == 200000
        assert len(calls) == (2 if mode == 'start' else 1)
        assert storage.load_torch(run / 'checkpoints/pilot-00100000.pt', full=True)['step'] == 100000
        assert storage.load_torch(run / 'checkpoints/pilot-00200000.pt', full=True)['step'] == 200000
        assert yaml.safe_load((run / 'config.yaml').read_text())['train']['steps'] == 500001
        snapshot = module.stage_root(run) / 'source-record.json'
        assert snapshot.read_bytes() == module.RECORD.read_bytes()
        stage_manifest = json.loads((module.stage_root(run) / 'stage-manifest.json').read_text())
        assert stage_manifest['source_record'] == str(snapshot)
        assert stage_manifest['source_record_sha256'] == module.io.sha(snapshot)
        record = json.loads(module.RECORD.read_text())
        record['phase'] = 'later_lifecycle_annotation'
        write_json(module.RECORD, record)
        monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Completed launch must not spawn'))
        assert module.launch_continuation(run)['already_completed']
    else:
        with pytest.raises((RuntimeError, ValueError)):
            module.main()
        assert json.loads(state_path.read_text())['state'] == 'failed'
        if mode not in ['child-failed', 'missing-validation']:
            assert not calls
        if mode == 'child-failed':
            result = module.launch_continuation(run)
            assert result['state'] == 'incomplete' and result['requires_inspection']
            assert len(calls) == 1


def test_detached_worker_keeps_both_locks_and_stale_launch_does_not_restart(continuation, tmp_path, monkeypatch):
    module = continuation
    run = tmp_path / 'pilot/seed-1'
    source_run(module, run)
    calls, descriptors = [], []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)
            descriptors.extend(os.dup(fd) for fd in kwargs.get('pass_fds', ()))

    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    try:
        assert module.launch_continuation(run)['pid'] == 12345
        assert module.launch_continuation(run)['already_running']
        with pytest.raises(BlockingIOError):
            module.pilot.acquire(module.pilot.GPU_LOCK)
        assert len(calls) == (2 if shutil.which('caffeinate') else 1)
    finally:
        for fd in descriptors:
            os.close(fd)
    assert module.launch_continuation(run)['requires_inspection']
    assert len(calls) == (2 if shutil.which('caffeinate') else 1)


def test_signal_is_forwarded_and_stops_queue(continuation, tmp_path, monkeypatch):
    module = continuation
    run = tmp_path / 'pilot/seed-1'
    source_run(module, run)
    monkeypatch.setattr(sys, 'argv', ['continue', '--run-dir', str(run)])
    calls, forwarded = [], []
    original = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(os, 'killpg', lambda pid, number: forwarded.append((pid, number)))

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)

        def poll(self):
            return None

        def wait(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    with pytest.raises(RuntimeError, match='stopped'):
        module.main()
    assert len(calls) == 1 and forwarded == [(12345, signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) == original
    assert json.loads((module.stage_root(run) / 'stage-state.json').read_text())['state'] == 'interrupted'


def test_environment_and_path_boundary(continuation, monkeypatch):
    with pytest.raises(ValueError, match='must stay inside'):
        continuation.launch_continuation(ROOT.parent / 'seed-1')
    monkeypatch.setattr(continuation, 'require_local_environment', lambda: (_ for _ in ()).throw(RuntimeError('local env required')))
    with pytest.raises(RuntimeError, match='local env required'):
        continuation.launch_continuation()


def test_corrupt_snapshot_is_not_replaced_by_live_record(continuation, tmp_path):
    run = tmp_path / 'pilot/seed-1'
    source_run(continuation, run)
    directory = continuation.stage_root(run)
    directory.mkdir(parents=True)
    snapshot = continuation.capture_source_record(directory)
    original_live = continuation.RECORD.read_bytes()
    snapshot.write_bytes(b'{incomplete json')
    with pytest.raises(ValueError):
        continuation.launch_continuation(run)
    assert snapshot.read_bytes() == b'{incomplete json'
    assert continuation.RECORD.read_bytes() == original_live


def test_snapshot_hash_changes_are_rejected(continuation, tmp_path):
    run = tmp_path / 'pilot/seed-1'
    source_run(continuation, run)
    directory = continuation.stage_root(run)
    directory.mkdir(parents=True)
    snapshot = continuation.capture_source_record(directory)
    expected = continuation.source_identity(run)
    write_json(directory / 'stage-manifest.json', expected)
    snapshot.write_bytes(snapshot.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='identity changed'):
        continuation.launch_continuation(run)
