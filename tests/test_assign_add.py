"""Protocol tests: leakage, split semantics, paired controls and deterministic resume."""
import copy
import math
import random

import pytest
import torch

from icl_lab.tasks.assign_add import GROUPS, Task


def configuration():
    return {"task_seed": 42,
            "data": {"modulus": 11, "n_vars": 6, "seq_len": 16,
                     "train_pair_fraction": 0.7, "forbidden_lhs": [4, 5], "forbidden_rhs": [0, 1]},
            "model": {"n_embd": 16, "n_head": 1, "n_layer": 2, "n_mlp": 32},
            "eval": {"diagnostic_episodes": 24}}


def decode_tokens(task, tokens):
    """Recover the operands from the *input alone*, without metadata/labels."""
    assignments = {}
    prefix = tokens[:-4].tolist()
    i = 0
    while i < len(prefix):
        if prefix[i] == task.pad:
            i += 1
            continue
        variable, value = prefix[i:i+2]
        assert task.first_var <= variable < task.first_var + task.n_vars
        assert 0 <= value < task.modulus
        assert variable not in assignments
        assignments[variable] = value
        i += 2
    assert tokens[-4] == task.plus and tokens[-1] == task.equal
    operands = [assignments[v] if v >= task.first_var else v for v in tokens[-3:-1].tolist()]
    return operands


def test_source_split_and_all_formats_obey_restrictions():
    task = Task(configuration())
    expected = set(random.Random(42).sample([(a, b) for a in range(11) for b in range(11)], math.floor(.7 * 121)))
    assert set(map(tuple, task.pair_mask.nonzero().tolist())) == expected
    batch = task.sample(512, torch.Generator().manual_seed(18))
    assert set(batch["type"].tolist()) == {0, 1, 2, 3}
    for i, tokens in enumerate(batch["tokens"]):
        operands = decode_tokens(task, tokens)
        assert tuple(operands) in expected
        assert sum(operands) % task.modulus == batch["target"][i]
        assert operands == batch["operands"][i].tolist()
        lhs, rhs = batch["variable_ids"][i].tolist()
        assert lhs not in task.forbidden_lhs and rhs not in task.forbidden_rhs
        if lhs >= 0 and rhs >= 0:
            assert lhs != rhs
        for j, pos in enumerate(batch["operand_positions"][i]):
            assert tokens[pos] == operands[j]


def test_all_held_out_groups_have_the_documented_membership():
    task = Task(configuration())
    batch = task.sample(120, torch.Generator().manual_seed(3), split="test")
    assert set(batch["group"].tolist()) == set(range(3, 9))
    for tokens, nums, names, group in zip(batch["tokens"], batch["operands"], batch["variable_ids"], batch["group"]):
        assert decode_tokens(task, tokens) == nums.tolist()
        seen = bool(task.pair_mask[nums[0], nums[1]])
        invalid = int(int(names[0]) in task.forbidden_lhs) + int(int(names[1]) in task.forbidden_rhs)
        if int(group) in (3, 4, 6):
            assert not seen and invalid == 0
        else:
            assert seen and invalid == (2 if group == 8 else 1)


def test_rng_and_static_state_restore_the_next_batch_exactly():
    task = Task(configuration())
    rng = torch.Generator().manual_seed(99)
    task.sample(13, rng)
    rng_state, task_state = rng.get_state(), copy.deepcopy(task.state_dict())
    expected = task.sample(31, rng)
    resumed = Task({**configuration(), "task_seed": 999})
    resumed.load_state_dict(task_state)
    actual = resumed.sample(31, torch.Generator().set_state(rng_state))
    for key in expected:
        assert torch.equal(expected[key], actual[key])


def test_model_only_reads_tokens_and_metrics_keep_empty_groups_explicit():
    torch.manual_seed(2)
    task = Task(configuration())
    model = task.build_model()
    batch = task.sample(16, torch.Generator().manual_seed(5))
    prediction = task.predict(model, batch)
    changed = {k: v.clone() for k, v in batch.items()}
    changed["target"].zero_()
    changed["operands"].zero_()
    changed["operand_positions"].zero_()
    assert torch.equal(prediction, task.predict(model, changed))
    loss = task.loss(prediction, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    metrics = task.metrics(prediction.detach(), batch)
    assert metrics["accuracy"].shape == (16,)
    assert torch.isnan(metrics["accuracy/0var_add_ood"]).all()
    assert not any("bias" in n for n, _ in model.named_parameters())
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())


def test_causal_attention_and_paired_diagnostics():
    torch.manual_seed(2)
    task = Task(configuration())
    model = task.build_model()
    batch = task.sample(24, torch.Generator().manual_seed(9), "test")
    cache = task.activations(model, batch)
    for name in ("attention1", "attention2"):
        assert torch.count_nonzero(cache[name].triu(diagonal=1)) == 0
    result = task.diagnostics(model, {"test": batch}, torch.device("cpu"))
    assert result["isolated_ov2_mlp2"]["all_pairs"] == 121
    assert result["test"]["n"] == 24
    assert set(result["test"]["by_group"]) == set(GROUPS[3:])


def test_invalid_sampling_and_config_fail_explicitly():
    with pytest.raises(ValueError):
        Task({**configuration(), "data": {**configuration()["data"], "train_pair_fraction": 1.0}})
    task = Task(configuration())
    with pytest.raises(ValueError):
        task.sample(1, torch.Generator(), "unknown")
