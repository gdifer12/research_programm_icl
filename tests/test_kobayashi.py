"""Protocol and information-flow checks; short training is not a paper reproduction."""
import copy
import json

import pytest
import torch

from icl_lab.tasks.kobayashi import Task, relative_position_bucket


def config(variant="vanilla"):
    return {"task_seed": 17, "data": {"n_dims": 4, "teacher_hidden": 4, "n_context": 8},
            "model": {"variant": variant, "n_embd": 16, "n_head": 4, "n_layer": 2, "hyper_hidden": 8},
            "eval": {"batch_size": 16, "diagnostic_episodes": 32},
            "analysis": {"ridge": [0.0, 0.01], "untrained_control": True, "shuffled_control": True}}


def bank(task, split="train", size=32, seed=10):
    return task.sample(size, torch.Generator().manual_seed(seed), split)


def test_connected_plus_split_and_latent_support():
    task = Task(config())
    train = bank(task, "train", 512)
    test = bank(task, "test", 128)
    assert len(task.train_pair_ids) == 12 and len(task.test_pair_ids) == 3
    assert not set(train["pair_id"].tolist()) & set(test["pair_id"].tolist())
    assert {tuple(p) for p in task.pairs[task.test_pair_ids].tolist()} == {(0, 3), (1, 4), (2, 5)}
    for batch in (train, test):
        assert torch.equal((batch["z"] > 0).float(), batch["mask"])
        assert torch.allclose(batch["z"].sum(-1), torch.full((len(batch["y"]),), 1.5))
        active = batch["z"][batch["mask"].bool()]
        assert (active >= 0.5).all() and (active <= 1).all()
        assert batch["x"].abs().max() <= 3**0.5


def test_teacher_formula_and_privileged_executor():
    task = Task(config())
    batch = bank(task)
    weights = torch.einsum("bm,mhd->bhd", batch["z"], task.theta)
    normalized = weights / torch.linalg.matrix_norm(weights, 2)[:, None, None]
    direct = torch.nn.functional.gelu(torch.einsum("bhd,bd->bh", normalized, batch["x"][:, -1])) @ task.readout
    assert torch.allclose(direct, batch["y"], atol=1e-7)
    oracle = task.baseline_predictions(batch)["privileged_true_code_executor"]
    assert torch.allclose(oracle, batch["y"], atol=1e-7)
    invalid, valid = task.teacher(torch.zeros_like(batch["z"]), batch["x"][:, -1])
    assert not valid.any() and torch.isnan(invalid).all()


def test_task_state_restores_teacher_and_does_not_consume_global_rng():
    torch.manual_seed(122)
    initial = torch.get_rng_state().clone()
    task = Task(config())
    assert torch.equal(initial, torch.get_rng_state())
    other_config = config()
    other_config["task_seed"] = 21
    other = Task(other_config)
    assert not torch.equal(task.theta, other.theta)
    other.load_state_dict(task.state_dict())
    for key, value in bank(task).items():
        assert torch.equal(value, bank(other)[key])


@pytest.mark.parametrize("variant", ["vanilla", "hypernetwork"])
def test_supervision_is_not_available_to_predictor_and_gradients_flow(variant):
    task = Task(config(variant))
    model = task.build_model()
    batch = bank(task)
    pred = task.predict(model, batch)
    tampered = dict(batch)
    tampered["z"] = torch.randn_like(batch["z"]) * 100
    tampered["y"] = torch.randn_like(batch["y"]) * 100
    tampered["mask"] = torch.zeros_like(batch["mask"])
    assert torch.equal(pred, task.predict(model, tampered))
    task.loss(pred, batch).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Query input can affect the answer; the context cannot expose its target.
    altered = dict(batch)
    altered["x"] = batch["x"].clone()
    altered["x"][:, -1] *= -1
    assert not torch.equal(pred, task.predict(model, altered))
    if variant == "hypernetwork":
        original_states = task.activations(model, batch)["last_token_residuals"]
        altered_states = task.activations(model, altered)["last_token_residuals"]
        assert torch.equal(original_states, altered_states)
        assert not any(torch.equal(p.detach(), task.theta) for p in model.parameters() if p.shape == task.theta.shape)


def test_noncausal_bidirectional_relative_positions():
    positions = torch.arange(10)
    buckets = relative_position_bucket(positions[None] - positions[:, None])
    assert torch.equal(buckets.diag(), torch.zeros(10, dtype=torch.long))
    assert (buckets.triu(1)[buckets.triu(1) != 0] >= 16).all()
    task = Task(config())
    model = task.build_model()
    batch = bank(task, size=2)
    tokens = torch.cat((batch["x"], torch.cat((batch["context_y"], torch.zeros(2, 1)), 1).unsqueeze(-1)), -1)
    embedded = model.read_in(tokens).detach().requires_grad_()
    output = model.blocks[0](embedded)
    gradient = torch.autograd.grad(output[:, 0].square().sum(), embedded)[0]
    assert gradient[:, -1].abs().sum() > 0  # Earlier token receives later information.


def test_probe_selects_only_on_id_validation_and_records_invalid_scores(monkeypatch):
    task = Task(config())
    model = task.build_model()
    banks = {split: bank(task, split, seed=offset + 30) for offset, split in enumerate(("probe_train", "validation", "id_test", "test"))}
    before = torch.get_rng_state().clone()
    mps_before = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    def reject_global_reseed(*args, **kwargs):
        pytest.fail("Probe diagnostics must not reseed global accelerator generators")
    monkeypatch.setattr(torch, "manual_seed", reject_global_reseed)
    diagnostics = task.diagnostics(model, banks, torch.device("cpu"))
    assert torch.equal(before, torch.get_rng_state())
    if mps_before is not None:
        assert torch.equal(mps_before, torch.mps.get_rng_state())
    json.dumps(diagnostics, allow_nan=False)
    assert diagnostics["selection_split"] == "validation (ID only)"
    assert {"trained_probe", "shuffled_label_probe", "untrained_model_probe"} <= diagnostics.keys()
    changed = copy.deepcopy(banks)
    changed["test"]["z"] *= 10
    changed_result = task.diagnostics(model, changed, torch.device("cpu"))
    original_fit = diagnostics["trained_probe"]
    changed_fit = changed_result["trained_probe"]
    for key in ("selected_layer", "selected_ridge", "coefficients"):
        assert original_fit[key] == changed_fit[key]
    assert original_fit["splits"]["test"]["invalid_executor_count"] == 0
    assert set(original_fit["splits"]["test"]["by_pair"]) == {str(i) for i in task.test_pair_ids.tolist()}
