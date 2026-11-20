"""Assign and Add, based on the pinned author protocol; independent PyTorch code.

The answer is never part of the input. Arithmetic-pair membership is ordered,
matching the author's code (not a symmetric split). Static split tensors are
checkpointed; all online draws derive solely from the supplied CPU generator.
"""
from __future__ import annotations

import math
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


GROUPS = (
    "0var_id", "1var_id", "2var_id", "0var_add_ood",
    "1var_add_ood", "1var_position_ood", "2var_add_ood",
    "2var_position_ood_1", "2var_position_ood_2",
)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("n_embd must be divisible by n_head")
        self.heads = heads
        self.head_width = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, capture: bool = False):
        b, length, width = x.shape
        def split(projection):
            return projection(x).view(b, length, self.heads, self.head_width).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        if capture:
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_width)
            causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            pattern = scores.masked_fill(~causal, float("-inf")).softmax(-1)
            result = pattern @ v
        else:
            pattern = None
            result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        result = self.out(result.transpose(1, 2).contiguous().view(b, length, width))
        return result, pattern

    def ov(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of all heads' OV contributions, before attention weighting."""
        return self.out(self.v(x))


class AssignAddModel(nn.Module):
    def __init__(self, vocab: int, modulus: int, seq_len: int, config: dict):
        super().__init__()
        width = int(config.get("n_embd", 128))
        heads = int(config.get("n_head", 1))
        if int(config.get("n_layer", 2)) != 2:
            raise ValueError("Assign and Add requires exactly two attention layers")
        self.embedding = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.empty(seq_len, width))
        self.attention1 = Attention(width, heads)
        self.attention2 = Attention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, int(config.get("n_mlp", 4 * width)), bias=False),
                                 nn.ReLU(),
                                 nn.Linear(int(config.get("n_mlp", 4 * width)), width, bias=False))
        self.readout = nn.Linear(width, modulus, bias=False)
        # Explicit fresh initialization. No equivalence to TransformerLens is claimed.
        std = float(config.get("init_std", 0.8 / math.sqrt(width)))
        for param in self.parameters():
            nn.init.normal_(param, mean=0.0, std=std)

    def forward(self, tokens: torch.Tensor, capture: bool = False):
        x = self.embedding(tokens) + self.position[:tokens.shape[1]]
        first, pattern1 = self.attention1(x, capture)
        after_first = x + first
        second, pattern2 = self.attention2(after_first, capture)
        before_mlp = after_first + second
        after_mlp = before_mlp + self.mlp(before_mlp)
        logits = self.readout(after_mlp[:, -1])
        if not capture:
            return logits
        return logits, {"attention1": pattern1, "attention2": pattern2,
                        "residual1": after_first, "pre_mlp": before_mlp,
                        "post_mlp": after_mlp}

    def isolated_addition(self, operands: torch.Tensor) -> torch.Tensor:
        # Published diagnostic: W_U MLP_2(OV_2(e_a) + OV_2(e_b)); no residual add.
        embeddings = self.embedding(operands)
        z = self.attention2.ov(embeddings).sum(dim=1)
        return self.readout(self.mlp(z))


class Task:
    name = "assign_add"
    group_names = GROUPS

    def __init__(self, config: dict):
        self.config = config
        self.data = config.get("data", {})
        self.modulus = int(self.data.get("modulus", 59))
        self.n_vars = int(self.data.get("n_vars", 12))
        self.seq_len = int(self.data.get("seq_len", 16))
        self.plus, self.equal, self.pad = self.modulus, self.modulus + 1, self.modulus + 2
        self.first_var = self.modulus + 3
        self.vars = list(range(self.n_vars))
        # Names reflect actual forbidden operand side, unlike author's legacy names.
        self.forbidden_rhs = list(self.data.get("forbidden_rhs", [0, 1]))
        self.forbidden_lhs = list(self.data.get("forbidden_lhs", [10, 11]))
        fraction = float(self.data.get("train_pair_fraction", 0.7))
        if self.modulus < 2 or self.n_vars < 4 or self.seq_len < 8:
            raise ValueError("modulus >= 2, n_vars >= 4 and seq_len >= 8 are required")
        for subset in (self.forbidden_lhs, self.forbidden_rhs):
            if not subset or len(set(subset)) != len(subset) or any(v not in self.vars for v in subset):
                raise ValueError("Forbidden-position indices must be distinct, nonempty valid variable IDs")
        if not 0 < fraction < 1:
            raise ValueError("train_pair_fraction must leave both training and held-out pairs")
        count = math.floor(fraction * self.modulus * self.modulus)
        if not 0 < count < self.modulus * self.modulus:
            raise ValueError("Empty arithmetic split")
        chosen = random.Random(int(config.get("task_seed", 42))).sample(
            [(a, b) for a in range(self.modulus) for b in range(self.modulus)], count)
        self.pair_mask = torch.zeros(self.modulus, self.modulus, dtype=torch.bool)
        for a, b in chosen:
            self.pair_mask[a, b] = True
        self._rebuild_pools()

    def _rebuild_pools(self):
        self.train_pairs = [tuple(v) for v in self.pair_mask.nonzero().tolist()]
        self.test_pairs = [tuple(v) for v in (~self.pair_mask).nonzero().tolist()]
        self.valid_lhs = [v for v in self.vars if v not in self.forbidden_lhs]
        self.valid_rhs = [v for v in self.vars if v not in self.forbidden_rhs]
        self.variable_pairs: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: []}
        for a in self.vars:
            for b in self.vars:
                if a != b:
                    bad = int(a in self.forbidden_lhs) + int(b in self.forbidden_rhs)
                    self.variable_pairs[bad].append((a, b))
        if not self.valid_lhs or not self.valid_rhs or any(not p for p in self.variable_pairs.values()):
            raise ValueError("Restrictions leave an empty variable-position evaluation group")
        # Counts of permissible author perm_table descriptors: type 0 has V rows,
        # type 1 excludes forbidden LHS, type 2 excludes RHS, type 3 keeps V-|LHS|.
        self.train_descriptors = ([(0, v) for v in self.vars] +
                                  [(1, v) for v in self.valid_lhs] +
                                  [(2, v) for v in self.valid_rhs] +
                                  [(3, v) for v in self.valid_lhs])

    def state_dict(self) -> dict:
        return {"pair_mask": self.pair_mask.clone(), "modulus": self.modulus,
                "n_vars": self.n_vars, "seq_len": self.seq_len,
                "forbidden_lhs": list(self.forbidden_lhs), "forbidden_rhs": list(self.forbidden_rhs)}

    def load_state_dict(self, state: dict):
        for key in ("modulus", "n_vars", "seq_len", "forbidden_lhs", "forbidden_rhs"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Task state/config mismatch: {key}")
        mask = state["pair_mask"].detach().cpu().bool()
        if mask.shape != (self.modulus, self.modulus) or not mask.any() or mask.all():
            raise ValueError("Invalid saved arithmetic split")
        self.pair_mask = mask.clone()
        self._rebuild_pools()

    def build_model(self) -> nn.Module:
        return AssignAddModel(self.first_var + self.n_vars, self.modulus, self.seq_len,
                              self.config.get("model", {}))

    def _row(self, rng: random.Random, split: str):
        if split != "test":
            typ, variable = rng.choice(self.train_descriptors)
            numbers = rng.choice(self.train_pairs)
            if typ == 0:
                names = (-1, -1)
            elif typ == 1:
                names = (variable, -1)
            elif typ == 2:
                names = (-1, variable)
            else:
                names = rng.choice(self.variable_pairs[0])
            return typ, names, numbers, variable, 0 if typ == 0 else 2 if typ == 3 else 1
        raise AssertionError("Test rows use explicit groups")

    def _test_row(self, rng: random.Random, group: int):
        held_out_pair = group in (3, 4, 6)
        numbers = rng.choice(self.test_pairs if held_out_pair else self.train_pairs)
        if group == 3:
            typ, names = 0, (-1, -1)
        elif group in (4, 5):
            typ = rng.choice((1, 2))
            pool = (self.forbidden_lhs if typ == 1 else self.forbidden_rhs) if group == 5 else (
                self.valid_lhs if typ == 1 else self.valid_rhs)
            variable = rng.choice(pool)
            names = (variable, -1) if typ == 1 else (-1, variable)
        else:
            typ = 3
            names = rng.choice(self.variable_pairs[{6: 0, 7: 1, 8: 2}[group]])
        return typ, names, numbers, rng.choice(self.vars), group

    def _tokens(self, rng: random.Random, typ: int, names: tuple, numbers: tuple, spare_var: int):
        assignments = [(v, numbers[side]) for side, v in enumerate(names) if v >= 0]
        if not assignments:
            assignments = [(spare_var, rng.randrange(self.modulus))]
        max_assignments = min(self.n_vars, (self.seq_len - 4) // 2)
        n_assignments = max(len(assignments), rng.randint(1, max_assignments))
        used = {v for v, _ in assignments}
        extras = [v for v in self.vars if v not in used]
        rng.shuffle(extras)
        assignments += [(v, rng.randrange(self.modulus)) for v in extras[:n_assignments-len(assignments)]]
        rng.shuffle(assignments)
        gaps = [0] * (n_assignments + 1)
        for _ in range(self.seq_len - 4 - 2 * n_assignments):
            gaps[rng.randrange(len(gaps))] += 1
        prefix, value_positions = [], {}
        for i, (v, value) in enumerate(assignments):
            prefix.extend([self.pad] * gaps[i])
            prefix.append(self.first_var + v)
            value_positions[v] = len(prefix)
            prefix.append(value)
        prefix.extend([self.pad] * gaps[-1])
        query = [self.first_var + v if v >= 0 else numbers[i] for i, v in enumerate(names)]
        positions = [value_positions[v] if v >= 0 else self.seq_len - 3 + i for i, v in enumerate(names)]
        return prefix + [self.plus] + query + [self.equal], positions

    def sample(self, batch_size: int, generator: torch.Generator, split: str = "train", step: int = 0):
        del step
        if split not in {"train", "probe_train", "validation", "id_test", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rng = random.Random(int(torch.randint(0, 2**63-1, (), generator=generator).item()))
        # A balanced, separately identified union of the six author's OOD groups.
        groups = [3 + i % 6 for i in range(batch_size)]
        rng.shuffle(groups)
        rows = [self._test_row(rng, group) if split == "test" else self._row(rng, split) for group in groups]
        tokens, targets, operands, names_all, positions, type_all, group_all = [], [], [], [], [], [], []
        for typ, names, nums, spare, group in rows:
            seq, pos = self._tokens(rng, typ, names, nums, spare)
            tokens.append(seq); targets.append(sum(nums) % self.modulus)
            operands.append(nums); names_all.append(names); positions.append(pos)
            type_all.append(typ); group_all.append(group)
        return {"tokens": torch.tensor(tokens, dtype=torch.long),
                "target": torch.tensor(targets, dtype=torch.long),
                "operands": torch.tensor(operands, dtype=torch.long),
                "variable_ids": torch.tensor(names_all, dtype=torch.long),
                "operand_positions": torch.tensor(positions, dtype=torch.long),
                "type": torch.tensor(type_all, dtype=torch.long),
                "group": torch.tensor(group_all, dtype=torch.long)}

    def predict(self, model, batch):
        return model(batch["tokens"])

    def loss(self, pred, batch):
        return F.cross_entropy(pred.float(), batch["target"])

    def metrics(self, pred, batch):
        correct = (pred.argmax(-1) == batch["target"]).float()
        out = {"accuracy": correct, "cross_entropy": F.cross_entropy(pred.float(), batch["target"], reduction="none")}
        for group, name in enumerate(GROUPS):
            out[f"accuracy/{name}"] = torch.where(batch["group"] == group, correct, torch.nan)
        for count in range(3):
            n = (batch["variable_ids"] >= 0).sum(-1)
            out[f"accuracy/{count}var"] = torch.where(n == count, correct, torch.nan)
        return out

    def baseline_predictions(self, batch):
        # Ground-truth arithmetic is privileged; label it explicitly, not a learned baseline.
        oracle = F.one_hot(batch["operands"].sum(-1) % self.modulus, self.modulus).float() * 100.0
        return {"oracle_arithmetic": oracle}

    def activations(self, model, batch):
        _, cache = model(batch["tokens"], capture=True)
        return cache

    @torch.no_grad()
    def diagnostics(self, model, banks: dict, device: torch.device):
        model.eval()
        limit = int(self.config.get("eval", {}).get("diagnostic_episodes", 256))
        output: dict[str, Any] = {"scope": "paired literal substitution, attention routing, isolated arithmetic; no causal attribution"}
        for split, source in banks.items():
            n = min(limit, len(source["target"]))
            batch = {k: v[:n].to(device) for k, v in source.items()}
            logits, cache = model(batch["tokens"], capture=True)
            literal = batch["tokens"].clone()
            literal[:, -3:-1] = batch["operands"]
            literal_logits = model(literal)
            real_correct = logits.argmax(-1) == batch["target"]
            literal_correct = literal_logits.argmax(-1) == batch["target"]
            positions = batch["operand_positions"]
            attention2 = cache["attention2"][:, :, -1, :].mean(1)
            attention1 = cache["attention1"][:, :, -1, :].mean(1)
            attention_at_operands = attention2.gather(1, positions).sum(1)
            output[split] = {
                "n": n, "model_accuracy": real_correct.float().mean().item(),
                "literal_accuracy": literal_correct.float().mean().item(),
                "literal_correct_model_wrong_fraction": (literal_correct & ~real_correct).float().mean().item(),
                "layer1_query_operand_attention": attention1[:, -3:-1].sum(1).mean().item(),
                "layer2_correct_value_attention": attention_at_operands.mean().item(),
                "by_group": {},
            }
            for i, name in enumerate(GROUPS):
                mask = batch["group"] == i
                if mask.any():
                    output[split]["by_group"][name] = {
                        "n": int(mask.sum()), "accuracy": real_correct[mask].float().mean().item(),
                        "literal_accuracy": literal_correct[mask].float().mean().item(),
                        "routing_attention": attention_at_operands[mask].mean().item(),
                    }
        all_pairs = torch.cartesian_prod(torch.arange(self.modulus), torch.arange(self.modulus))
        correct_parts = []
        for pairs in all_pairs.split(512):
            pairs = pairs.to(device)
            correct_parts.append((model.isolated_addition(pairs).argmax(-1) == pairs.sum(-1) % self.modulus).cpu())
        correct = torch.cat(correct_parts)
        seen = self.pair_mask.flatten()
        output["isolated_ov2_mlp2"] = {"all_pairs": len(all_pairs),
                                        "accuracy": correct.float().mean().item(),
                                        "seen_accuracy": correct[seen].float().mean().item(),
                                        "held_out_accuracy": correct[~seen].float().mean().item()}
        return output
