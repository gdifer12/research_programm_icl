import shutil
"""Pilot boundaries and process ownership without GPU workloads."""
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


@pytest.fixture
def pilot(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('garg_pilot', ROOT / 'scripts/garg_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.identity
    monkeypatch.setattr(module, 'identity', lambda: {**original(), 'code_hash': 'test-code'})
    monkeypatch.setattr(storage, 'code_hash', lambda: 'test-code')
    monkeypatch.setattr(module, 'GPU_LOCK', tmp_path / '.gpu.lock')
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def saved_run(pilot, run, step=50000):
    run.mkdir(parents=True, exist_ok=True)
    config = configuration.load(pilot.CONFIG, seed=1)
    (run / 'config.yaml').write_text(yaml.safe_dump(config))
    write_json(run / 'manifest.json', {'code_hash': 'test-code'})
    write_json(run / 'status.json', {'state': 'paused', 'step': step})
    checkpoint = run / 'checkpoints/last.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b'checkpoint fixture')
    bank = run / 'banks/validation.npz'
    bank.parent.mkdir(exist_ok=True)
    bank.write_bytes(b'bank fixture')
    write_json(run / 'banks/manifest.json', {'banks': {'validation': {'file': bank.name, 'sha256': pilot.io.sha(bank)}}})


def saved_eval(pilot, run):
    directory = run / 'evaluations/validation-00050000-fixture'
    directory.mkdir(parents=True, exist_ok=True)
    predictions = directory / 'predictions.npz'
    predictions.write_bytes(b'prediction fixture')
    write_json(directory / 'summary.json', {'step': 50000, 'split': 'validation', 'code_hash': 'test-code',
               'checkpoint_sha256': pilot.io.sha(run / 'checkpoints/last.pt'),
               'bank_sha256': pilot.io.sha(run / 'banks/validation.npz'), 'predictions': str(predictions)})


@pytest.mark.parametrize('mode', ['fresh', 'paused-at', 'paused-below', 'paused-above', 'changed-config',
                                 'changed-code', 'child-failed', 'missing-validation'])
def test_worker_stops_at_pilot_boundary(pilot, tmp_path, monkeypatch, mode):
    run = tmp_path / 'pilot/seed-1'
    monkeypatch.setattr(sys, 'argv', ['pilot', '--run-dir', str(run)])
    if mode not in ['fresh', 'child-failed']:
        saved_run(pilot, run, step={'paused-below': 49999, 'paused-above': 50001}.get(mode, 50000))
    if mode == 'changed-config':
        config = yaml.safe_load((run / 'config.yaml').read_text())
        config['train']['steps'] = 50000
        (run / 'config.yaml').write_text(yaml.safe_dump(config))
    if mode == 'changed-code':
        write_json(run / 'manifest.json', {'code_hash': 'other-code'})
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
                assert '--steps' not in self.command and '--resume' not in self.command
                saved_run(pilot, run)
            elif mode != 'missing-validation':
                assert self.command[self.command.index('--split') + 1] == 'validation'
                assert self.command[self.command.index('--checkpoint') + 1] == 'last'
                saved_eval(pilot, run)
            return 0

    monkeypatch.setattr(pilot.subprocess, 'Popen', Child)
    if mode in ['fresh', 'paused-at']:
        pilot.main()
        assert len(calls) == (2 if mode == 'fresh' else 1)
        state = json.loads((run.parent / 'pilot-state.json').read_text())
        assert state['state'] == 'pilot_completed' and state['step'] == 50000
        assert yaml.safe_load((run / 'config.yaml').read_text())['train']['steps'] == 500001
    else:
        with pytest.raises(RuntimeError):
            pilot.main()
        assert json.loads((run.parent / 'pilot-state.json').read_text())['state'] == 'failed'
        if mode in ['paused-below', 'paused-above', 'changed-config', 'changed-code']:
            assert not calls


def test_launch_is_idempotent_and_does_not_restart_stale_worker(pilot, tmp_path, monkeypatch):
    run = tmp_path / 'pilot/seed-1'
    held, calls = [], []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)
            held.extend(os.dup(fd) for fd in kwargs.get('pass_fds', ()))

    monkeypatch.setattr(pilot.subprocess, 'Popen', Child)
    try:
        assert pilot.launch_pilot(run)['pid'] == 12345
        assert pilot.launch_pilot(run)['already_running']
        assert len(calls) == (2 if shutil.which('caffeinate') else 1)
    finally:
        for fd in held:
            os.close(fd)
    assert pilot.launch_pilot(run)['requires_inspection']
    assert len(calls) == (2 if shutil.which('caffeinate') else 1)


def test_completed_launch_reuses_validation_and_rejects_stale_checkpoint(pilot, tmp_path, monkeypatch):
    run = tmp_path / 'pilot/seed-1'
    saved_run(pilot, run)
    saved_eval(pilot, run)
    write_json(run.parent / 'pilot-manifest.json', {**pilot.identity(), 'saved_config_sha256': pilot.io.sha(run / 'config.yaml')})
    write_json(run.parent / 'pilot-state.json', {'state': 'pilot_completed', 'pid': 12345})
    monkeypatch.setattr(pilot.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must not launch'))
    assert pilot.launch_pilot(run)['already_completed']
    (run / 'checkpoints/last.pt').write_bytes(b'other checkpoint branch')
    assert pilot.validation_summary(run, pilot.identity()) is None


def test_paused_below_boundary_requires_inspection(pilot, tmp_path, monkeypatch):
    run = tmp_path / 'pilot/seed-1'
    saved_run(pilot, run, step=1000)
    monkeypatch.setattr(pilot.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must not resume'))
    result = pilot.launch_pilot(run)
    assert result['state'] == 'incomplete' and result['requires_inspection']


def test_signal_forwarding_stops_before_evaluation(pilot, tmp_path, monkeypatch):
    run = tmp_path / 'pilot/seed-1'
    monkeypatch.setattr(sys, 'argv', ['pilot', '--run-dir', str(run)])
    forwarded, calls = [], []
    monkeypatch.setattr(os, 'killpg', lambda pid, sig: forwarded.append((pid, sig)))
    original = signal.getsignal(signal.SIGTERM)

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)

        def poll(self):
            return None

        def wait(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return 0

    monkeypatch.setattr(pilot.subprocess, 'Popen', Child)
    with pytest.raises(RuntimeError, match='stopped'):
        pilot.main()
    assert len(calls) == 1 and forwarded == [(12345, signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) == original
    assert json.loads((run.parent / 'pilot-state.json').read_text())['state'] == 'interrupted'


def test_external_paths_are_rejected(pilot):
    with pytest.raises(ValueError, match='must stay inside'):
        pilot.resolve(ROOT.parent / 'seed-1')
