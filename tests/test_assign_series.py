"""Verify resume and failure boundaries without launching a GPU workload."""
import importlib.util
import json
from pathlib import Path
import signal
import sys

import pytest
import yaml

ROOT = Path(__file__).parents[1]


@pytest.fixture
def worker(monkeypatch):
    spec = importlib.util.spec_from_file_location('assign_series', ROOT / 'scripts/run-assign-series.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep this test independent of ongoing edits to scripts during a real run.
    monkeypatch.setattr(module, 'code_hash', lambda: 'test-code')
    handlers = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM]}
    yield module
    for number, handler in handlers.items():
        signal.signal(number, handler)


@pytest.mark.parametrize('mode', ['completed', 'paused', 'missing-final', 'changed-config'])
def test_series_respects_completion_and_final_evaluations(worker, tmp_path, monkeypatch, mode):
    run = tmp_path / 'seed-1'
    config = worker.configuration.load(ROOT / 'configs/experiments/assign_add.yaml', seed=1)
    monkeypatch.setattr(sys, 'argv', ['run-assign-series.py', '--run-root', str(tmp_path), '--seeds', '1'])
    calls = []
    if mode != 'paused':
        run.mkdir()
        saved = {**config, 'model_seed': 999} if mode == 'changed-config' else config
        (run / 'config.yaml').write_text(yaml.safe_dump(saved))
        (run / 'manifest.json').write_text(json.dumps({'code_hash': 'test-code'}))
        (run / 'status.json').write_text(json.dumps({'state': 'completed', 'step': 30000}))

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            self.command = command
            calls.append(command)
            assert kwargs['pass_fds'], 'An orphan child must keep the series lock'

        def wait(self):
            if 'train' in self.command:
                run.mkdir(exist_ok=True)
                (run / 'status.json').write_text(json.dumps({'state': 'paused', 'step': 10}))
            else:
                split = self.command[self.command.index('--split') + 1]
                directory = run / 'evaluations' / f'{split}-00030000-test'
                directory.mkdir(parents=True, exist_ok=True)
                predictions = directory / 'predictions.npz'
                predictions.write_bytes(b'test fixture')
                (directory / 'summary.json').write_text(json.dumps({
                    'step': 0 if mode == 'missing-final' else 30000, 'predictions': str(predictions)}))
            return 0

    monkeypatch.setattr(worker.subprocess, 'Popen', Child)
    if mode == 'completed':
        worker.main()
        worker.main()  # Running the same notebook again must never retrain completed seeds.
        assert all('evaluate' in command for command in calls)
        assert len(calls) == 4
        assert json.loads((tmp_path / 'series-state.json').read_text())['state'] == 'completed'
    else:
        with pytest.raises(RuntimeError):
            worker.main()
        assert json.loads((tmp_path / 'series-state.json').read_text())['state'] == 'failed'
        if mode == 'paused':
            assert len(calls) == 1 and 'train' in calls[0]
        elif mode == 'changed-config':
            assert not calls
        else:
            assert len(calls) == 1


def test_series_reads_reject_paths_outside_project(worker):
    with pytest.raises(ValueError, match='must stay inside'):
        worker.read_json(ROOT.parent / 'outside.json')
