"""Post-hoc CPU interventions on the three saved Assign and Add models. Print JSON."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from icl_lab.tasks.assign_add import Task


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/assign-add-jupyter-20261119"
RANDOM_SEED = 20261120


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(model, tokens):
    return torch.cat([model(batch).argmax(-1) for batch in tokens.split(128)])


def audit_ood(task, bank, predicted):
    # Decode the two names from literal token pairs, ignoring the labels and
    # lookup metadata until after the independent answer has been calculated.
    lengths = []
    for i, row in enumerate(bank["tokens"]):
        values = {}
        for pos, token in enumerate(row[:12]):
            if token >= task.first_var:
                assert token not in values and row[pos + 1] < task.modulus
                values[token] = pos + 1
        addresses = [values[row[pos]] if row[pos] >= task.first_var else pos for pos in (13, 14)]
        operands = row[addresses]
        assert np.array_equal(addresses, bank["operand_positions"][i])
        assert np.array_equal(operands, bank["operands"][i])
        assert sum(operands) % task.modulus == bank["target"][i]
        lengths.append(len(values))
    lengths = np.array(lengths)
    correct = predicted == bank["target"]
    results = {}
    for group in (7, 8):
        selected = bank["group"] == group
        by_length = [correct[selected & (lengths == n)].mean() for n in range(2, 7)]
        results[task.group_names[group]] = {
            "empirical_accuracy": float(correct[selected].mean()),
            "uniform_length_accuracy": float(np.mean(by_length))}
    return {"n_decoded": len(lengths), "two_variable_position_groups": results}


def remove_distractors(model, bank, task):
    # Begin with full six-assignment episodes. Deleting a (name, value) pair
    # preserves absolute positions, total length, literal, target and query.
    tokens = torch.from_numpy(bank["tokens"])
    full = (tokens[:, :12] >= task.first_var).sum(1) == 6
    rng = random.Random(RANDOM_SEED)
    results = {}
    for n_var in (1, 2):
        mask = full & torch.tensor((bank["variable_ids"] >= 0).sum(1) == n_var)
        original = tokens[mask]
        labels = torch.from_numpy(bank["target"])[mask]
        current = original.clone()
        orders = []
        for i, row in enumerate(original):
            extra = [p for p in range(0, 12, 2) if row[p] not in row[-3:-1]]
            assert len(extra) == 6 - n_var
            rng.shuffle(extra)
            orders.append(extra)
            for pos in extra:
                current[i, pos:pos+2] = task.pad
        orders = torch.tensor(orders)
        ids = torch.arange(len(current))
        counts = [int(predict(model, current).eq(labels).sum())]
        for positions in orders.T:
            for col in (positions, positions + 1):
                current[ids, col] = original[ids, col]
            counts.append(int(predict(model, current).eq(labels).sum()))
        assert torch.equal(current, original)
        results[f"{n_var}_variable"] = {"n": len(current), "correct_by_extra_count": counts}
    return results


def make_challenge(task, seen, n=1200):
    """Six distinct assignments; two admissible queries with unequal answers.

    The two intended sums have the same train-pair membership. A/B differ only
    by the name in the query. Balance left/right exactly, with no padding.
    """
    rng = random.Random(RANDOM_SEED + int(seen))
    queries, addresses, answers = [], [], []
    while len(queries) < n:
        side = len(queries) % 2
        names = rng.sample(task.vars, 6)
        values = rng.sample(range(task.modulus), 6)
        literal = rng.randrange(task.modulus)
        valid_names = task.valid_lhs if side == 0 else task.valid_rhs
        allowed = []
        for i, (name, value) in enumerate(zip(names, values)):
            pair = (value, literal) if side == 0 else (literal, value)
            if name in valid_names and bool(task.pair_mask[pair]) == seen:
                allowed.append(i)
        if len(allowed) < 2:
            continue
        a, b = rng.sample(allowed, 2)
        prefix = [token for name, value in zip(names, values)
                  for token in (task.first_var + name, value)]
        rows = []
        for index in (a, b):
            operands = [literal, literal]
            operands[side] = task.first_var + names[index]
            rows.append(prefix + [task.plus, *operands, task.equal])
        queries.append(rows)
        addresses.append((2*a + 1, 2*b + 1, 13 + (1-side)))
        answers.append(((values[a] + literal) % task.modulus,
                        (values[b] + literal) % task.modulus))
    return torch.tensor(queries), torch.tensor(addresses), torch.tensor(answers)


def challenge_scores(model, challenge):
    queries, addresses, answers = challenge
    rows_a, rows_b = queries.unbind(1)
    native_a = predict(model, rows_a)
    native_b = predict(model, rows_b)
    expected_a, expected_b = answers.unbind(1)
    counts = {"n_pairs": len(queries),
              "natural_a_correct": int((native_a == expected_a).sum()),
              "natural_b_correct": int((native_b == expected_b).sum()),
              "both_queries_correct": int(((native_a == expected_a) & (native_b == expected_b)).sum()),
              "unchanged_answer_on_query_swap": int((native_a == native_b).sum())}
    forced = {"correct_address_target": 0, "wrong_address_target": 0, "wrong_address_decoy": 0}
    target_is_top_value = 0
    for tokens, positions, labels, baseline in zip(rows_a.split(128), addresses.split(128),
                                                  answers.split(128), native_a.split(128)):
        logits, cache = model(tokens, capture=True)
        assert torch.equal(logits.argmax(-1), baseline)
        ids = torch.arange(len(tokens))
        target, decoy, literal = positions.T
        attention = cache["attention2"][:, 0, -1]
        top_value = 2 * attention[:, 1:12:2].argmax(-1) + 1
        target_is_top_value += int((top_value == target).sum())
        residual = cache["residual1"]

        def routed(value_pos):
            # Replace only the row of layer-2 attention read by the answer.
            # Keep values, weights, residual branches and MLP unchanged.
            mixed = 0.5 * (residual[ids, value_pos] + residual[ids, literal])
            pre_mlp = residual[:, -1] + model.attention2.ov(mixed)
            return model.readout(pre_mlp + model.mlp(pre_mlp)).argmax(-1)

        right, wrong = routed(target), routed(decoy)
        forced["correct_address_target"] += int((right == labels[:, 0]).sum())
        forced["wrong_address_target"] += int((wrong == labels[:, 0]).sum())
        forced["wrong_address_decoy"] += int((wrong == labels[:, 1]).sum())
    return {**counts, "target_top_among_six_value_positions": target_is_top_value, **forced}


def make_name_test(task):
    # Enumerate all unequal training pairs; a single assignment can begin in
    # any of the 11 legal slots. Reuse the very same positions for every name.
    pairs = torch.tensor([(a, b) for a, b in task.train_pairs if a != b])
    rng = random.Random(RANDOM_SEED)
    slots = torch.tensor([sum(rng.randrange(2) for _ in range(10)) for _ in pairs])
    result = {}
    ids = torch.arange(len(pairs))
    for side, names in [(0, "cdkl"), (1, "abcd")]:
        for name in names:
            token = task.first_var + ord(name) - ord("a")
            literal = pairs[:, 1-side]
            query = torch.full((len(pairs), task.seq_len), task.pad)
            query[ids, slots] = token
            query[ids, slots + 1] = pairs[:, side]
            query[:, -4] = task.plus
            query[:, -3+side] = token
            query[:, -3+(1-side)] = literal
            query[:, -1] = task.equal
            result[(side, name)] = query, pairs.sum(1) % task.modulus, 2*literal % task.modulus
    return result


def name_scores(model, examples):
    result = {}
    for (side, name), (tokens, target, doubled) in examples.items():
        pred = predict(model, tokens)
        assert not (target == doubled).any()
        result[f"{name}_{'left' if side == 0 else 'right'}"] = {
            "n": len(pred), "correct": int((pred == target).sum()),
            "doubled_literal": int((pred == doubled).sum())}
    return result


def patch_name_attention(model, examples):
    # Same checkpoint, same episode. Only the variable's name differs. Keep
    # recipient values, residual branches and MLP. The '=' token is last, so
    # every key is visible. This assay is for the published single-head model.
    assert model.attention2.heads == 1
    bad, labels, _ = examples[1, "a"]
    good, _, _ = examples[1, "c"]
    control = good.roll(1, dims=0)
    attention = model.attention2
    counts = {}
    for a, c, shuffled, target in zip(bad.split(96), good.split(96), control.split(96), labels.split(96)):
        def residual(tokens):
            return model(tokens, capture=True)[1]["residual1"]

        ra, rc, rs = map(residual, (a, c, shuffled))
        name_in_prefix = a[:, :12] == a[:, -2, None]
        assert torch.all(name_in_prefix.sum(1) == 1)
        value_positions = name_in_prefix.long().argmax(1) + 1
        ids = torch.arange(len(a))
        def read(recipient, query_source, key_source):
            q = attention.q(query_source[:, -1:])
            k = attention.k(key_source)
            weights = ((q @ k.transpose(1, 2)) / attention.head_width**0.5).softmax(-1)
            output = attention.out(weights @ attention.v(recipient))[:, 0]
            pre_mlp = recipient[:, -1] + output
            return model.readout(pre_mlp + model.mlp(pre_mlp)).argmax(-1)

        for name, recipient, qs, ks in [
            ("native_a", ra, ra, ra), ("native_c", rc, rc, rc),
            ("a_with_c_attention", ra, rc, rc), ("c_with_a_attention", rc, ra, ra),
            ("a_with_unpaired_c_attention", ra, rs, rs),
            ("a_with_c_query_only", ra, rc, ra), ("a_with_c_keys_only", ra, ra, rc),
        ]:
            pred = read(recipient, qs, ks)
            if name in ("native_a", "native_c"):
                assert torch.equal(pred, predict(model, a if name == "native_a" else c))
            counts[name] = counts.get(name, 0) + int((pred == target).sum())
        for name, position in [("assigned_value", value_positions), ("assigned_name", value_positions-1),
                               ("query_name", 14), ("literal", 13), ("equals", 15)]:
            for prefix, recipient, donor in [("a_with_c", ra, rc), ("c_with_a", rc, ra)]:
                keys = recipient.clone()
                keys[ids, position] = donor[ids, position]
                pred = read(recipient, recipient, keys)
                key = f"{prefix}_key_at_{name}"
                counts[key] = counts.get(key, 0) + int((pred == target).sum())
    return {"n": len(labels), "correct": counts}


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    results = {"protocol": "paired distractor deletion and name/address interventions",
               "run": str(RUN.relative_to(ROOT)), "device": "cpu", "torch": str(torch.__version__),
               "challenge_rng_seed": RANDOM_SEED, "seeds": {}}
    for seed in (1, 2, 3):
        folder = RUN / f"seed-{seed}"
        task = Task(yaml.safe_load((folder / "config.yaml").read_text()))
        weights = folder / "checkpoints/final.pt"
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        assert checkpoint["step"] == 30000
        task.load_state_dict(checkpoint["task_state"])
        model = task.build_model().eval()
        model.load_state_dict(checkpoint["model"])
        saved, = folder.glob("evaluations/test-00030000-*/predictions.npz")
        bank_path = folder / "banks/test.npz"
        with np.load(bank_path) as bank, np.load(saved) as preds:
            assert np.array_equal(bank["episode_id"], preds["episode_id"])
            native = predict(model, torch.from_numpy(bank["tokens"])).numpy()
            agree = int((native == preds["prediction"].argmax(1)).sum())
            assert agree == len(native)
            result = {"checkpoint_sha256": sha256(weights), "bank_sha256": sha256(bank_path),
                      "mps_argmax_matched": agree, "ood_audit": audit_ood(task, bank, native),
                      "deletion": remove_distractors(model, bank, task)}
        for name, seen in (("new_numeric_pairs", False), ("familiar_numeric_pairs", True)):
            result[name] = challenge_scores(model, make_challenge(task, seen))
        name_test = make_name_test(task)
        result["single_assignment_all_unequal_seen_pairs"] = name_scores(model, name_test)
        result["name_attention_patch"] = patch_name_attention(model, name_test)
        # Compare training trajectories at identical fixed examples; do not pick
        # checkpoints by test performance. Checkpoint selection is exploratory.
        trajectory = {}
        familiar, _, answers = make_challenge(task, seen=True, n=400)
        for step in (1000, 3000, 5000, 10000, 20000, 30000):
            path = folder / f"checkpoints/step-{step:08d}.pt"
            weights_at_step = torch.load(path, map_location="cpu", weights_only=True)
            assert weights_at_step["step"] == step
            model.load_state_dict(weights_at_step["model"])
            trajectory[step] = {
                "checkpoint_sha256": sha256(path),
                "six_assignments_correct_out_of_400": int((predict(model, familiar[:, 0]) == answers[:, 0]).sum()),
                "single_assignment": name_scores(model, {k: v for k, v in name_test.items()
                                                         if k in [(1, "b"), (1, "c")]})}
        result["trajectory"] = trajectory
        results["seeds"][seed] = result
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
