import shutil
"""Exercise series ownership, resume, and readiness with mocked child processes."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import pytest
import yaml

ROOT = Path(__file__).parents[1]


@pytest.fixture
def worker(monkeypatch):
    spec = importlib.util.spec_from_file_location('kobayashi_series', ROOT / 'scripts/run-kobayashi-series.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_identity = module.series.identity
    monkeypatch.setattr(module.series, 'identity',
                        lambda variant, seeds: {**original_identity(variant, seeds), 'code_hash': 'test-code'})
    monkeypatch.setattr(module, 'code_hash', lambda: 'test-code')
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def saved_run(worker, root, seed=1, state='completed'):
    run = root / f'seed-{seed}'
    run.mkdir(parents=True, exist_ok=True)
    config = worker.configuration.load(ROOT / 'configs/experiments' / worker.series.CONFIGS[root.name], seed=seed)
    (run / 'config.yaml').write_text(yaml.safe_dump(config))
    write_json(run / 'manifest.json', {'code_hash': 'test-code'})
    write_json(run / 'status.json', {'state': state, 'step': 100000 if state == 'completed' else 10})
    checkpoint = run / 'checkpoints/step-00100000.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b'final model fixture')
    banks = {}
    for split in ['id_test', 'test']:
        bank = run / 'banks' / f'{split}.npz'
        bank.parent.mkdir(exist_ok=True)
        bank.write_bytes(split.encode())
        banks[split] = {'file': bank.name, 'sha256': worker.series.sha(bank)}
    write_json(run / 'banks/manifest.json', {'banks': banks})
    return run


def saved_eval(worker, run, split, step=100000):
    directory = run / 'evaluations' / f'{split}-00100000-fixture'
    directory.mkdir(parents=True, exist_ok=True)
    predictions = directory / 'predictions.npz'
    predictions.write_bytes(b'prediction fixture')
    write_json(directory / 'summary.json', {
        'step': step, 'split': split, 'code_hash': 'test-code', 'predictions': str(predictions),
        'checkpoint_sha256': worker.series.sha(run / 'checkpoints/step-00100000.pt'),
        'bank_sha256': worker.series.sha(run / 'banks' / f'{split}.npz')})


def saved_series(worker, root, seeds=(1,)):
    expected = worker.series.identity(root.name, seeds)
    write_json(root / 'series-manifest.json', expected)
    write_json(root / 'series-state.json', {'state': 'completed', 'pid': 123})
    for seed in seeds:
        run = saved_run(worker, root, seed)
        for split in ['id_test', 'test']:
            saved_eval(worker, run, split)
    return expected


@pytest.mark.parametrize('mode', ['fresh', 'completed', 'resume', 'paused', 'missing-final',
                                 'changed-config', 'changed-code'])
def test_worker_full_series_and_failures(worker, tmp_path, monkeypatch, mode):
    root = tmp_path / 'vanilla'
    seeds = [1, 2, 3] if mode == 'fresh' else [1]
    monkeypatch.setattr(sys, 'argv', ['worker', '--variant', 'vanilla', '--run-root', str(root),
                                    '--seeds', *map(str, seeds)])
    if mode not in ['fresh', 'paused']:
        run = saved_run(worker, root, state='paused' if mode == 'resume' else 'completed')
        if mode == 'changed-config':
            config_path = run / 'config.yaml'
            config = yaml.safe_load(config_path.read_text())
            config['model_seed'] = 999
            config_path.write_text(yaml.safe_dump(config))
        if mode == 'changed-code':
            write_json(run / 'manifest.json', {'code_hash': 'old-code'})
    calls = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            self.command = command
            calls.append(command)
            assert len(kwargs['pass_fds']) == 2
            assert kwargs['start_new_session']
            for descriptor in kwargs['pass_fds']:
                os.fstat(descriptor)

        def wait(self):
            run = Path(self.command[self.command.index('--run-dir') + 1])
            if 'train' in self.command:
                saved_run(worker, root, int(run.name.split('-')[1]),
                          state='paused' if mode == 'paused' else 'completed')
            else:
                split = self.command[self.command.index('--split') + 1]
                assert self.command[self.command.index('--checkpoint') + 1] == 'all'
                saved_eval(worker, run, split, step=10 if mode == 'missing-final' else 100000)
            return 0

    monkeypatch.setattr(worker.subprocess, 'Popen', Child)
    if mode in ['fresh', 'completed', 'resume']:
        worker.main()
        assert json.loads((root / 'series-state.json').read_text())['state'] == 'completed'
        assert len(calls) == (9 if mode == 'fresh' else 3 if mode == 'resume' else 2)
        if mode == 'fresh':
            assert [Path(c[c.index('--run-dir') + 1]).name for c in calls] == [f'seed-{s}' for s in seeds for _ in range(3)]
        if mode == 'resume':
            assert '--resume' in calls[0] and '--config' not in calls[0]
        if mode == 'completed':
            assert all('evaluate' in command for command in calls)
    else:
        with pytest.raises(RuntimeError):
            worker.main()
        assert json.loads((root / 'series-state.json').read_text())['state'] == 'failed'
        if mode in ['changed-config', 'changed-code']:
            assert not calls
        elif mode == 'paused':
            assert len(calls) == 1 and 'train' in calls[0]


def test_signal_stops_queue_and_is_forwarded(worker, tmp_path, monkeypatch):
    root = tmp_path / 'vanilla'
    monkeypatch.setattr(sys, 'argv', ['worker', '--variant', 'vanilla', '--run-root', str(root)])
    forwarded, calls = [], []
    monkeypatch.setattr(os, 'killpg', lambda pid, number: forwarded.append((pid, number)))
    previous = signal.getsignal(signal.SIGTERM)

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)

        def poll(self):
            return None

        def wait(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return 0

    monkeypatch.setattr(worker.subprocess, 'Popen', Child)
    with pytest.raises(RuntimeError, match='stopped'):
        worker.main()
    assert len(calls) == 1
    assert forwarded == [(12345, signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) == previous
    assert json.loads((root / 'series-state.json').read_text())['state'] == 'interrupted'


def test_worker_requires_vanilla_and_records_early_failure(worker, tmp_path, monkeypatch):
    root = tmp_path / 'hypernetwork'
    monkeypatch.setattr(sys, 'argv', ['worker', '--variant', 'hypernetwork', '--run-root', str(root)])
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must not launch'))
    with pytest.raises(RuntimeError, match='completed vanilla'):
        worker.main()
    assert json.loads((root / 'series-state.json').read_text())['state'] == 'failed'


def test_launcher_is_idempotent_and_holds_shared_lock(worker, tmp_path, monkeypatch):
    calls, held = [], []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)
            if kwargs.get('pass_fds'):
                assert len(kwargs['pass_fds']) == 2
                held.extend(os.dup(descriptor) for descriptor in kwargs['pass_fds'])

    monkeypatch.setattr(worker.series.subprocess, 'Popen', Child)
    try:
        launched = worker.series.launch_series('vanilla', run_root=tmp_path / 'vanilla')
        assert launched['pid'] == 12345 and launched['state'] == 'running'
        again = worker.series.launch_series('vanilla', run_root=tmp_path / 'vanilla')
        assert again['already_running']
        blocked = worker.series.launch_series('hypernetwork', run_root=tmp_path / 'hypernetwork')
        assert blocked['state'] == 'blocked'
        assert len(calls) == (2 if shutil.which('caffeinate') else 1)
    finally:
        for descriptor in held:
            os.close(descriptor)


def test_completed_launch_reuses_series_and_unlocks_hyper_dependency(worker, tmp_path, monkeypatch):
    root = tmp_path / 'vanilla'
    saved_series(worker, root)
    monkeypatch.setattr(worker.series.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must not retrain'))
    reused = worker.series.launch_series('vanilla', seeds=(1,), run_root=root)
    assert reused['already_completed'] and reused['state'] == 'completed'
    worker.series.check_dependency(tmp_path / 'hypernetwork', worker.series.identity('hypernetwork', [1]))


@pytest.mark.parametrize('changed', ['checkpoint', 'bank', 'predictions'])
def test_completed_series_rejects_stale_or_missing_artifacts(worker, tmp_path, changed):
    root = tmp_path / 'vanilla'
    expected = saved_series(worker, root)
    run = root / 'seed-1'
    if changed == 'checkpoint':
        (run / 'checkpoints/step-00100000.pt').write_bytes(b'other training branch')
    elif changed == 'bank':
        (run / 'banks/test.npz').write_bytes(b'other bank')
    else:
        (run / 'evaluations/test-00100000-fixture/predictions.npz').unlink()
    assert not worker.series.completed(root, expected)
    with pytest.raises(RuntimeError, match='completed vanilla'):
        worker.series.check_dependency(tmp_path / 'hypernetwork', worker.series.identity('hypernetwork', [1]))


def test_boundaries_reject_external_symlink_before_traversal(worker, tmp_path):
    run = saved_run(worker, tmp_path / 'vanilla')
    (run / 'evaluations').mkdir()
    (run / 'evaluations/id_test-escape').symlink_to(ROOT.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='must stay inside'):
        worker.series.final_evaluations_exist(run, worker.series.identity('vanilla', [1]))
    with pytest.raises(ValueError, match='must stay inside'):
        worker.series.resolve_root('vanilla', ROOT.parent / 'vanilla')


def test_worker_rejects_shared_lock_without_overwriting_state(worker, tmp_path, monkeypatch):
    root = tmp_path / 'vanilla'
    write_json(root / 'series-state.json', {'state': 'running', 'pid': 987})
    monkeypatch.setattr(sys, 'argv', ['worker', '--variant', 'vanilla', '--run-root', str(root)])
    with (tmp_path / '.gpu-series.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already holds'):
            worker.main()
    assert json.loads((root / 'series-state.json').read_text()) == {'state': 'running', 'pid': 987}
