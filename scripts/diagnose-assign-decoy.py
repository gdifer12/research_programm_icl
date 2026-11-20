"""Test a predicted decoy against a matched decoy on fresh Assign and Add inputs.

From the checkout, use ./scripts/local-python scripts/diagnose-assign-decoy.py. The only inputs
on disk are the three final checkpoints and their run configs. All counterfactual
inputs stay in memory and the measurements are printed as JSON.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from icl_lab.tasks.assign_add import Task


EPISODE_SEED = 20261120803
EDIT_SEED = 20261120804
EPISODES = 3000
MODULUS = 59
ARITHMETIC_OOD_ONE_VARIABLE = 4


def load_checkpoint(runs, seed):
    run = runs / f"seed-{seed}"
    task = Task(yaml.safe_load((run / "config.yaml").read_text()))
    saved = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=True)
    task.load_state_dict(saved["task_state"])
    model = task.build_model()
    model.load_state_dict(saved["model"])
    return task, model.eval()


@torch.inference_mode()
def predict(model, sequences):
    return torch.cat([model(batch).argmax(-1) for batch in sequences.split(128)]).numpy()


def fresh_episodes(task):
    rng = random.Random(EPISODE_SEED)
    sequences, metadata = [], []
    while len(sequences) < EPISODES:
        typ, names, operands, spare, _ = task._test_row(rng, ARITHMETIC_OOD_ONE_VARIABLE)
        tokens, positions = task._tokens(rng, typ, names, operands, spare)
        if sum(token >= task.first_var for token in tokens[:12]) != 6:
            continue
        side = 0 if typ == 1 else 1
        # With six assignments the 12 prefix tokens alternate names and numbers.
        target_assignment = positions[side] // 2
        sequences.append(tokens)
        metadata.append((side, operands, target_assignment))
    return sequences, metadata


def paired_edits(task, sequences, metadata, before):
    rng = random.Random(EDIT_SEED)
    pair_mask = task.pair_mask.numpy()
    edited_source, edited_control = [], []
    selected_indices, truth, increments = [], [], []
    baseline_correct = 0
    wrong_matching_decoy = 0
    for index, ((side, operands, target), tokens, prediction) in enumerate(
        zip(metadata, sequences, before)
    ):
        prediction = int(prediction)
        correct_sum = sum(operands) % MODULUS
        literal = operands[1 - side]
        values = tokens[1:12:2]
        assert values[target] == operands[side]
        if prediction == correct_sum:
            baseline_correct += 1
            continue
        implied = (prediction - literal) % MODULUS
        if implied not in values:
            continue
        wrong_matching_decoy += 1
        if len(set(values)) != 6:
            continue
        source = values.index(implied)
        assert source != target

        def is_seen(value):
            pair = (value, literal) if side == 0 else (literal, value)
            return bool(pair_mask[pair])

        # Preserve enumeration order: the edit RNG draws uniformly from eligible
        # (control, delta) pairs. Known/held-out status matches between the two
        # branches before the edit and also after it; it may change across time.
        choices = []
        for control in range(6):
            if control in (target, source) or is_seen(values[control]) != is_seen(implied):
                continue
            for delta in range(1, MODULUS):
                new_source = (implied + delta) % MODULUS
                new_control = (values[control] + delta) % MODULUS
                if new_source in values or new_control in values:
                    continue
                if is_seen(new_source) == is_seen(new_control):
                    choices.append((control, delta))
        if not choices:
            continue
        control, delta = rng.choice(choices)
        source_tokens = tokens.copy()
        control_tokens = tokens.copy()
        source_tokens[2 * source + 1] = (implied + delta) % MODULUS
        control_tokens[2 * control + 1] = (values[control] + delta) % MODULUS
        edited_source.append(source_tokens)
        edited_control.append(control_tokens)
        selected_indices.append(index)
        truth.append(correct_sum)
        increments.append(delta)
    return (
        torch.tensor(edited_source),
        torch.tensor(edited_control),
        np.array(selected_indices),
        np.array(truth),
        np.array(increments),
        baseline_correct,
        wrong_matching_decoy,
    )


def main():
    runs = Path(__file__).resolve().parents[1] / "artifacts/runs/assign-add-jupyter-20261119"
    torch.set_num_threads(2)
    task, model = load_checkpoint(runs, 2)
    sequences, metadata = fresh_episodes(task)
    original = torch.tensor(sequences)
    baseline = predict(model, original)
    edited, control, selected, truth, delta, correct, hits = paired_edits(
        task, sequences, metadata, baseline
    )
    initial = baseline[selected]
    assert np.array_equal(initial, predict(model, original[selected]))
    after = predict(model, edited)
    placebo = predict(model, control)
    follows = after == (initial + delta) % MODULUS
    control_follows = placebo == (initial + delta) % MODULUS
    results = {
        "episode_rng_seed": EPISODE_SEED,
        "edit_rng_seed": EDIT_SEED,
        "episodes": EPISODES,
        "baseline_correct": int(correct),
        "wrong_matching_decoy": int(hits),
        "selected": len(selected),
        "follows_delta_source_control": [int(follows.sum()), int(control_follows.sum())],
        "source_only_control_only_both": [
            int(np.sum(follows & ~control_follows)),
            int(np.sum(control_follows & ~follows)),
            int(np.sum(follows & control_follows)),
        ],
        "unchanged_source_control": [int(np.sum(after == initial)), int(np.sum(placebo == initial))],
        "correct_source_control": [int(np.sum(after == truth)), int(np.sum(placebo == truth))],
        "other_training_seeds": {},
    }
    for seed in (1, 3):
        _, strong_model = load_checkpoint(runs, seed)
        base = predict(strong_model, original[selected])
        changed = predict(strong_model, edited)
        other = predict(strong_model, control)
        results["other_training_seeds"][seed] = {
            "correct_original_source_control": [int(np.sum(x == truth)) for x in (base, changed, other)],
            "follows_own_delta_source_control": [
                int(np.sum(x == (base + delta) % MODULUS)) for x in (changed, other)
            ],
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
