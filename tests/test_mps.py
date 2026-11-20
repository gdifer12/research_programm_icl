"""Backend contracts plus short real-Metal lifecycle checks on Apple Silicon."""
from pathlib import Path
import copy
import json
import pytest
import torch
from icl_lab import config
from icl_lab.devices import require_device
from icl_lab.engine import train, rng_state, restore_rng, evaluate, load_run
from icl_lab.storage import load_torch, save_torch

ROOT = Path(__file__).parents[1]
MPS_AVAILABLE = torch.backends.mps.is_built() and torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not MPS_AVAILABLE, reason='Native Metal access required')


def small(name):
    c = config.load(ROOT / f'configs/experiments/{name}.yaml', smoke=True)
    c['train'].update(steps=4, save_every=2, eval_every=2, log_every=1)
    c['eval'].update(episodes=12, diagnostic_episodes=12, batch_size=6, bootstrap_samples=10)
    return c


def test_unavailable_mps_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CPU fallback is disabled'):
        require_device('mps')


def test_mps_rejects_implicit_fallback_and_fp16(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_built', lambda: True)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.raises(ValueError, match='implicit CPU fallback'):
        require_device('mps')
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    with pytest.raises(ValueError, match='requires fp32'):
        require_device('mps', 'fp16')


@requires_mps
def test_mps_rng_roundtrip():
    generator = torch.Generator().manual_seed(19)
    torch.mps.manual_seed(31)
    saved = rng_state(generator, 'mps')
    expected = torch.rand(13, device='mps').cpu()
    torch.rand(29, device='mps')
    restore_rng(saved, generator, 'mps')
    assert torch.equal(expected, torch.rand(13, device='mps').cpu())
    broken = copy.deepcopy(saved); broken['mps'] = None
    with pytest.raises(ValueError, match='missing its device RNG'):
        restore_rng(broken, generator, 'mps')


@requires_mps
@pytest.mark.parametrize('name', ['garg', 'assign_add', 'kobayashi', 'kobayashi_hyper'])
def test_mps_resume_and_evaluation(name, tmp_path):
    c = small(name)
    whole, part = tmp_path / 'whole', tmp_path / 'part'
    train(c, whole, device='mps')
    train(c, part, device='mps', stop_after=2)
    train(c, part, device='mps', resume='last')
    a, b = [load_torch(path / 'checkpoints/last.pt', full=True) for path in (whole, part)]
    assert a['training_device'] == b['training_device'] == 'mps'
    assert torch.equal(a['rng']['train_generator'], b['rng']['train_generator'])
    assert torch.equal(a['rng']['mps'], b['rng']['mps'])
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=1e-5, atol=1e-6)
    result = evaluate(part, device='mps')
    assert result['step'] == 4
    assert result == evaluate(part, device='mps')
    assert next(load_run(part, device='mps')[2].parameters()).device.type == 'mps'
    rows = [json.loads(line) for line in (part / 'metrics.jsonl').read_text().splitlines()]
    assert rows and all(row['update_seconds'] > 0 and row['device'] == 'mps' for row in rows)
    assert all(row['allocated_bytes'] > 0 and row['driver_allocated_bytes'] > 0 for row in rows)


@requires_mps
def test_cross_backend_resume_requires_explicit_flag(tmp_path):
    c = small('assign_add'); run = tmp_path / 'run'
    train(c, run, device='cpu', stop_after=2)
    with pytest.raises(ValueError, match='allow-device-change'):
        train(c, run, device='mps', resume='last')
    train(c, run, device='mps', resume='last', allow_device_change=True)
    state = load_torch(run / 'checkpoints/last.pt', full=True)
    assert state['training_device'] == 'mps' and state['rng']['mps'] is not None
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    start = [event for event in events if event.get('event') == 'start'][-1]
    assert start['device_changed'] and start['prior_training_device'] == 'cpu'


def test_legacy_checkpoint_uses_manifest_device(tmp_path):
    c = small('assign_add'); run = tmp_path / 'run'
    train(c, run, device='cpu', stop_after=2)
    path = run / 'checkpoints/last.pt'
    saved = load_torch(path, full=True)
    saved.pop('training_device')
    saved['rng'].pop('mps')
    save_torch(path, saved)
    train(c, run, device='cpu', resume='last')
    assert load_torch(path, full=True)['step'] == 4
