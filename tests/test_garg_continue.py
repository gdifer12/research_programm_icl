import shutil
"""Verify the one authorized 50k -> 100k stage with CPU fixtures and mocked workers."""
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
    spec = importlib.util.spec_from_file_location('garg_continue', ROOT / 'scripts/garg_continue.py')
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
    write_json(run / 'status.json', {'state': 'paused', 'step': 50000})
    checkpoint = run / 'checkpoints/pilot-00050000.pt'
    storage.save_torch(checkpoint, payload(config, 50000))
    os.link(checkpoint, run / 'checkpoints/last.pt')
    (run / 'checkpoints/last.pt.sha256').write_text(module.io.sha(checkpoint))
    write_json(run.parent / 'pilot-manifest.json', {**module.pilot.identity(),
               'config_hash': configuration.digest(config), 'saved_config_sha256': module.io.sha(run / 'config.yaml')})
    write_json(module.RECORD, {'run_dir': str(run), 'seed': 1, 'pilot_target_steps': 50000,
               'full_config_steps': 500001, 'device': 'mps', 'precision': 'fp32', 'technical_smoke': False,
               'code_hash': 'test-code', 'config_hash': configuration.digest(config),
               'preserved_checkpoint': {'path': str(checkpoint), 'sha256': module.io.sha(checkpoint), 'resumable': True}})
    bank = run / 'banks/validation.npz'
    bank.parent.mkdir()
    bank.write_bytes(b'bank fixture')
    write_json(run / 'banks/manifest.json', {'banks': {'validation': {'file': bank.name, 'sha256': module.io.sha(bank)}}})
    return config


def final_state(run, config, step=100000):
    storage.save_torch(run / 'checkpoints/last.pt', payload(config, step))
    write_json(run / 'status.json', {'state': 'paused', 'step': step})


def validation(module, run):
    directory = run / 'evaluations/validation-00100000-fixture'
    directory.mkdir(parents=True, exist_ok=True)
    prediction = directory / 'predictions.npz'
    prediction.write_bytes(b'prediction fixture')
    write_json(directory / 'summary.json', {'step': 100000, 'split': 'validation', 'code_hash': 'test-code',
               'checkpoint_sha256': module.io.sha(run / 'checkpoints/pilot-00100000.pt'),
               'bank_sha256': module.io.sha(run / 'banks/validation.npz'), 'predictions': str(prediction)})


@pytest.mark.parametrize('mode', ['start', 'target', 'below', 'above', 'changed-config', 'changed-code',
                                 'source-corrupted', 'last-advanced', 'child-failed', 'missing-validation', 'partial-pair'])
def test_authorized_budget_and_failure_boundaries(continuation, tmp_path, monkeypatch, mode):
    module = continuation
    run = tmp_path / 'pilot/seed-1'
    config = source_run(module, run)
    monkeypatch.setattr(sys, 'argv', ['continue', '--run-dir', str(run)])
    if mode in ['target', 'missing-validation', 'partial-pair']:
        final_state(run, config)
    if mode in ['below', 'above']:
        final_state(run, config, 75000 if mode == 'below' else 100001)
    if mode == 'partial-pair':
        os.link(run / 'checkpoints/last.pt', run / 'checkpoints/pilot-00100000.pt')
    if mode == 'changed-config':
        changed = {**config, 'model_seed': 2}
        (run / 'config.yaml').write_text(yaml.safe_dump(changed))
    if mode == 'changed-code':
        monkeypatch.setattr(storage, 'code_hash', lambda: 'other-code')
    if mode == 'source-corrupted':
        (run / 'checkpoints/pilot-00050000.pt').write_bytes(b'corruption')
    if mode == 'last-advanced':
        storage.save_torch(run / 'checkpoints/last.pt', payload(config, 50001))
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
                assert self.command[self.command.index('--stop-after') + 1] == '50000'
                assert Path(self.command[self.command.index('--resume') + 1]) == run / 'checkpoints/pilot-00050000.pt'
                assert not {'--config', '--seed', '--steps', '--allow-code-change', '--allow-device-change'} & set(self.command)
                final_state(run, config)
            elif mode != 'missing-validation':
                assert self.command[self.command.index('--split') + 1] == 'validation'
                assert Path(self.command[self.command.index('--checkpoint') + 1]) == run / 'checkpoints/pilot-00100000.pt'
                validation(module, run)
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    state_path = module.stage_root(run) / 'stage-state.json'
    if mode in ['start', 'target', 'partial-pair']:
        module.main()
        state = json.loads(state_path.read_text())
        assert state['state'] == 'stage_completed' and state['step'] == 100000
        assert len(calls) == (2 if mode == 'start' else 1)
        assert storage.load_torch(run / 'checkpoints/pilot-00050000.pt', full=True)['step'] == 50000
        assert storage.load_torch(run / 'checkpoints/pilot-00100000.pt', full=True)['step'] == 100000
        assert yaml.safe_load((run / 'config.yaml').read_text())['train']['steps'] == 500001
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
