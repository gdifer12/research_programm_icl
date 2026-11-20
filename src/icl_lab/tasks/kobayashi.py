"""Kobayashi modular regression: paper-based reconstruction, not author code.

The task contains privileged labels/teacher state for scoring. Models receive only
x and context_y. See experiments/kobayashi-implementation.md for fixed choices.
"""
from __future__ import annotations

import copy
import itertools
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def relative_position_bucket(relative: Tensor, buckets: int = 32, max_distance: int = 128) -> Tensor:
    """Bidirectional T5-style buckets; relative = key position - query position."""
    if buckets < 4 or buckets % 2 or max_distance <= buckets // 4:
        raise ValueError("Use an even bucket count >=4 and max_distance > buckets/4")
    half = buckets // 2
    distance = relative.abs()
    exact = half // 2
    large = exact + (
        torch.log(distance.float().clamp_min(exact) / exact)
        / math.log(max_distance / exact)
        * (half - exact)
    ).long()
    local = torch.where(distance < exact, distance, large.clamp_max(half - 1))
    return local + (relative > 0).long() * half


class RelativeAttention(nn.Module):
    def __init__(self, width: int, heads: int, buckets: int, max_distance: int, bias: bool):
        super().__init__()
        if width % heads:
            raise ValueError("n_embd must be divisible by n_head")
        self.heads = heads
        self.head_dim = width // heads
        self.buckets = buckets
        self.max_distance = max_distance
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        self.relative_bias = nn.Embedding(buckets, heads)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        positions = torch.arange(length, device=x.device)
        bucket = relative_position_bucket(
            positions[None, :] - positions[:, None], self.buckets, self.max_distance
        )
        bias = self.relative_bias(bucket).permute(2, 0, 1).unsqueeze(0)
        # Deliberately noncausal: Appendix C.4 says "Attention mask: None".
        attn = ((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias).softmax(-1)
        values = (attn @ v).transpose(1, 2).reshape(batch, length, width)
        return self.proj(values)


class Block(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        width = int(cfg.get("n_embd", 128))
        bias = bool(cfg.get("bias", True))
        self.norm1 = nn.LayerNorm(width, eps=float(cfg.get("norm_eps", 1e-5)))
        self.norm2 = nn.LayerNorm(width, eps=float(cfg.get("norm_eps", 1e-5)))
        self.attention = RelativeAttention(
            width, int(cfg.get("n_head", 4)), int(cfg.get("relative_buckets", 32)),
            int(cfg.get("relative_max_distance", 128)), bias,
        )
        hidden = int(cfg.get("expansion", 4)) * width
        self.mlp = nn.Sequential(nn.Linear(width, hidden, bias=bias), nn.GELU(), nn.Linear(hidden, width, bias=bias))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ModularRegressor(nn.Module):
    def __init__(self, n_dims: int, cfg: dict[str, Any]):
        super().__init__()
        self.variant = str(cfg.get("variant", "vanilla"))
        if self.variant not in {"vanilla", "hypernetwork"}:
            raise ValueError("Kobayashi variant must be vanilla or hypernetwork")
        self.n_dims = n_dims
        width = int(cfg.get("n_embd", 128))
        depth = int(cfg.get("n_layer", 2))
        if depth < 1:
            raise ValueError("n_layer must be positive")
        self.read_in = nn.Linear(n_dims + 1, width, bias=bool(cfg.get("bias", True)))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width) if cfg.get("final_norm", False) else nn.Identity()
        if self.variant == "vanilla":
            self.read_out = nn.Linear(width, 1, bias=bool(cfg.get("bias", True)))
        else:
            self.latent_dim = int(cfg.get("latent_dim", 6))
            self.hyper_hidden = int(cfg.get("hyper_hidden", 32))
            self.to_latent = nn.Linear(width, self.latent_dim, bias=False)
            # Learned modules and learned readout, never copied from the teacher.
            self.modules_weight = nn.Parameter(torch.empty(self.latent_dim, self.hyper_hidden, n_dims))
            self.hyper_readout = nn.Parameter(torch.empty(self.hyper_hidden))
        self.apply(self._initialize)
        if self.variant == "hypernetwork":
            nn.init.normal_(self.modules_weight, std=1 / math.sqrt(self.latent_dim * n_dims))
            nn.init.normal_(self.hyper_readout, std=1 / math.sqrt(self.hyper_hidden))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, context_y: Tensor, return_activations: bool = False):
        if x.ndim != 3 or context_y.shape != (x.shape[0], x.shape[1] - 1):
            raise ValueError("Expected x[B,K+1,D] and context_y[B,K]")
        contexts = torch.cat((x[:, :-1], context_y.unsqueeze(-1)), dim=-1)
        query_token = torch.zeros(x.shape[0], 1, self.n_dims + 1, device=x.device, dtype=x.dtype)
        if self.variant == "vanilla":
            query_token[:, 0, :self.n_dims] = x[:, -1]
        # Hypernetwork code inference has no access to the query x or query y.
        tokens = torch.cat((contexts, query_token), dim=1)
        hidden = self.read_in(tokens)
        states = [hidden[:, -1]] if return_activations else None
        for block in self.blocks:
            hidden = block(hidden)
            if return_activations:
                states.append(hidden[:, -1])
        final = self.final_norm(hidden[:, -1])
        latent = None
        if self.variant == "vanilla":
            pred = self.read_out(final).squeeze(-1)
        else:
            latent = self.to_latent(final)
            weights = torch.einsum("bm,mhd->bhd", latent, self.modules_weight)
            values = F.gelu(torch.einsum("bhd,bd->bh", weights, x[:, -1]))
            pred = values @ self.hyper_readout
        if return_activations:
            return pred, torch.stack(states, dim=1), latent
        return pred


def _truncated_normal(shape: tuple[int, ...], generator: torch.Generator, std: float,
                      bound: float, rescale: bool) -> Tensor:
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("teacher_truncation must be finite and positive")
    low = 0.5 * (1 + math.erf(-bound / math.sqrt(2)))
    high = 1 - low
    u = torch.rand(shape, generator=generator) * (high - low) + low
    value = torch.erfinv(2 * u - 1) * math.sqrt(2)
    if rescale:
        variance = 1 - 2 * bound * math.exp(-bound * bound / 2) / math.sqrt(2 * math.pi) / (high - low)
        value = value / math.sqrt(variance)
    return value * std


class Task:
    """Fixed teacher with independently sampled episodes and Connected+ masks."""
    def __init__(self, config: dict[str, Any]):
        self.config = copy.deepcopy(config)
        data = self.config.get("data", {})
        self.n_dims = int(data.get("n_dims", 16))
        self.teacher_hidden = int(data.get("teacher_hidden", 16))
        self.n_context = int(data.get("n_context", 64))
        self.n_modules = int(data.get("n_modules", 6))
        if self.n_modules != 6:
            raise ValueError("Connected+ protocol requires exactly six modules")
        if min(self.n_dims, self.teacher_hidden, self.n_context) < 1:
            raise ValueError("Dimensions and n_context must be positive")
        self.pairs = torch.tensor(list(itertools.combinations(range(6), 2)), dtype=torch.long)
        held_out = {(0, 3), (1, 4), (2, 5)}
        self.test_pair_ids = torch.tensor([i for i, pair in enumerate(self.pairs.tolist()) if tuple(pair) in held_out])
        self.train_pair_ids = torch.tensor([i for i, pair in enumerate(self.pairs.tolist()) if tuple(pair) not in held_out])
        generator = torch.Generator().manual_seed(int(self.config.get("task_seed", 17)))
        trunc = float(data.get("teacher_truncation", 2.0))
        rescale = bool(data.get("teacher_std_rescale", True))
        self.theta = _truncated_normal((6, self.teacher_hidden, self.n_dims), generator, 1 / math.sqrt(6), trunc, rescale)
        self.readout = _truncated_normal((self.teacher_hidden,), generator, 1 / math.sqrt(self.teacher_hidden), trunc, rescale)

    def state_dict(self) -> dict[str, Any]:
        return {"protocol": "kobayashi-connected-plus-v1", "theta": self.theta.clone(),
                "readout": self.readout.clone(), "pairs": self.pairs.clone(),
                "train_pair_ids": self.train_pair_ids.clone(), "test_pair_ids": self.test_pair_ids.clone()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("protocol") != "kobayashi-connected-plus-v1":
            raise ValueError("Unknown teacher protocol")
        for name in ("theta", "readout", "pairs", "train_pair_ids", "test_pair_ids"):
            value = state[name].detach().cpu().clone()
            reference = getattr(self, name)
            if value.shape != reference.shape:
                raise ValueError(f"Task state shape mismatch for {name}")
            if name not in {"theta", "readout"} and not torch.equal(value, reference):
                raise ValueError(f"Task split mismatch for {name}")
            if name in {"theta", "readout"} and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite teacher state: {name}")
            setattr(self, name, value)

    def teacher(self, z: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """Privileged executor. Invalid raw codes produce NaN, never silent filtering."""
        theta = self.theta.to(device=x.device, dtype=x.dtype)
        readout = self.readout.to(device=x.device, dtype=x.dtype)
        z = z.to(device=x.device, dtype=x.dtype)
        weights = torch.einsum("bm,mhd->bhd", z, theta)
        finite = torch.isfinite(weights).flatten(1).all(1)
        safe = torch.where(finite[:, None, None], weights, torch.zeros_like(weights))
        norm = torch.linalg.matrix_norm(safe, ord=2, dim=(-2, -1))
        valid = finite & torch.isfinite(norm) & (norm > 1e-12)
        normalized = safe / norm.clamp_min(1e-12)[:, None, None]
        query_only = x.ndim == 2
        inputs = x[:, None] if query_only else x
        hidden = torch.einsum("bhd,bkd->bkh", normalized, inputs)
        y = F.gelu(hidden) @ readout
        y = torch.where(valid[:, None], y, torch.full_like(y, float("nan")))
        return (y[:, 0] if query_only else y), valid

    def sample(self, batch_size: int, generator: torch.Generator, split: str = "train", step: int = 0) -> dict[str, Tensor]:
        del step
        if split not in {"train", "probe_train", "validation", "id_test", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        ids = self.test_pair_ids if split == "test" else self.train_pair_ids
        pair_id = ids[torch.randint(len(ids), (batch_size,), generator=generator)]
        mask = torch.zeros(batch_size, 6)
        mask.scatter_(1, self.pairs[pair_id], 1.0)
        exponential = -torch.log(torch.rand(batch_size, 6, generator=generator).clamp_min(torch.finfo(torch.float32).tiny)) * mask
        z = 0.5 * (exponential / exponential.sum(-1, keepdim=True) + mask)
        x = (torch.rand(batch_size, self.n_context + 1, self.n_dims, generator=generator) * 2 - 1) * math.sqrt(3)
        y, valid = self.teacher(z, x)
        if not valid.all():
            raise ValueError("Generated teacher has an invalid operator norm")
        return {"x": x, "context_y": y[:, :-1], "y": y[:, -1], "z": z,
                "mask": mask, "pair_id": pair_id}

    def build_model(self) -> nn.Module:
        return ModularRegressor(self.n_dims, self.config.get("model", {}))

    def predict(self, model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
        return model(batch["x"], batch["context_y"])

    def loss(self, pred: Tensor, batch: dict[str, Tensor]) -> Tensor:
        return F.mse_loss(pred, batch["y"])

    def metrics(self, pred: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        error = (pred - batch["y"]).square()
        baseline_error = (batch["context_y"].mean(-1) - batch["y"]).square()
        result = {"mse": error, "context_mean_mse": baseline_error}
        for pair_id in range(len(self.pairs)):
            keep = batch["pair_id"] == pair_id
            result[f"pair_{pair_id:02d}_mse"] = error.masked_fill(~keep, float("nan"))
            result[f"pair_{pair_id:02d}_context_mean_mse"] = baseline_error.masked_fill(~keep, float("nan"))
        return result

    def baseline_predictions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        truth, _ = self.teacher(batch["z"], batch["x"][:, -1])
        return {"context_mean": batch["context_y"].mean(-1), "zero": torch.zeros_like(batch["y"]),
                "privileged_true_code_executor": truth}

    def activations(self, model: nn.Module, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        pred, states, latent = model(batch["x"], batch["context_y"], return_activations=True)
        result = {"last_token_residuals": states, "prediction": pred}
        if latent is not None:
            result["learned_bottleneck"] = latent
        return result

    def _features(self, model: nn.Module, batch: dict[str, Tensor], device: torch.device) -> Tensor:
        chunk = int(self.config.get("eval", {}).get("batch_size", 128))
        outputs = []
        with torch.no_grad():
            for offset in range(0, len(batch["y"]), chunk):
                x = batch["x"][offset:offset + chunk].to(device)
                context_y = batch["context_y"][offset:offset + chunk].to(device)
                _, states, _ = model(x, context_y, return_activations=True)
                outputs.append(states.detach().cpu().double())
        return torch.cat(outputs)

    def diagnostics(self, model: nn.Module, banks: dict[str, dict[str, Tensor]], device: torch.device) -> dict[str, Any]:
        required = {"probe_train", "validation", "id_test", "test"}
        if not required <= set(banks):
            raise ValueError(f"Probe requires banks {sorted(required)}")
        options = self.config.get("analysis", {})
        count = int(self.config.get("eval", {}).get("diagnostic_episodes", 1024))
        selected = {key: {name: tensor[:count].cpu() for name, tensor in banks[key].items()} for key in sorted(required)}
        old_training = model.training
        model.eval()
        try:
            features = {key: self._features(model, batch, device) for key, batch in selected.items()}
            result: dict[str, Any] = {
                "protocol": "full-six-coordinate-ridge-id-validation-v1",
                "selection_split": "validation (ID only)",
                "feature_semantics": "query-inclusive last-token residual" if getattr(model, "variant", "vanilla") == "vanilla" else "context-only zero-token residual",
                "episodes": {key: len(batch["y"]) for key, batch in selected.items()},
                "trained_probe": self._probe_report(features, selected),
            }
            comparisons = {}
            chunk = int(self.config.get("eval", {}).get("batch_size", 128))
            for split in ("validation", "id_test", "test"):
                batch = selected[split]
                parts = []
                with torch.no_grad():
                    for offset in range(0, len(batch["y"]), chunk):
                        inputs = {key: value[offset:offset + chunk].to(device) for key, value in batch.items()}
                        parts.append(self.predict(model, inputs).detach().cpu().double())
                prediction = torch.cat(parts)
                error = (prediction - batch["y"].double()).square()
                probe_split = result["trained_probe"]["splits"][split]
                executor_mse = probe_split["executor_mse"]
                comparisons[split] = {
                    "model_mse": float(error.mean()),
                    "executor_mse": executor_mse,
                    "mean_model_minus_executor_squared_error": float(error.mean()) - executor_mse if executor_mse is not None else None,
                    "model_prediction": prediction.tolist(),
                    "by_pair": {
                        str(pid): {"count": int((batch["pair_id"] == pid).sum()),
                                   "model_mse": float(error[batch["pair_id"] == pid].mean()),
                                   "executor_mse": probe_split["by_pair"][str(pid)]["executor_mse"]}
                        for pid in torch.unique(batch["pair_id"]).tolist()
                    },
                }
            result["model_vs_privileged_executor"] = comparisons
            if bool(options.get("shuffled_control", True)):
                generator = torch.Generator().manual_seed(int(options.get("control_seed", 7301)))
                perm = torch.randperm(len(selected["probe_train"]["z"]), generator=generator)
                shuffled = dict(selected)
                shuffled["probe_train"] = dict(selected["probe_train"])
                shuffled["probe_train"]["z"] = selected["probe_train"]["z"][perm]
                result["shuffled_label_probe"] = self._probe_report(features, shuffled)
            if bool(options.get("untrained_control", True)):
                with torch.random.fork_rng(devices=[]):
                    # The control is initialized on CPU. torch.manual_seed also
                    # reseeds accelerator RNGs, which this CPU-only fork does not
                    # restore; keep diagnostics isolated from MPS/CUDA streams.
                    torch.random.default_generator.manual_seed(int(options.get("control_seed", 7301)) + 1)
                    untrained = self.build_model().to(device).eval()
                controls = {key: self._features(untrained, batch, device) for key, batch in selected.items()}
                result["untrained_model_probe"] = self._probe_report(controls, selected)
                del untrained
            return result
        finally:
            model.train(old_training)

    def _probe_report(self, features: dict[str, Tensor], banks: dict[str, dict[str, Tensor]]) -> dict[str, Any]:
        lambdas = [float(value) for value in self.config.get("analysis", {}).get("ridge", [0.0, 1e-4, 1e-2, 1.0])]
        if not lambdas or min(lambdas) < 0:
            raise ValueError("Ridge grid must be nonempty and nonnegative")
        target = banks["probe_train"]["z"].double()
        best = None
        # Layers and regularization selected on ID validation, never OOD.
        for layer in range(features["probe_train"].shape[1]):
            train_x = features["probe_train"][:, layer]
            mean = train_x.mean(0)
            scale = train_x.std(0, unbiased=False).clamp_min(1e-8)
            train_x = (train_x - mean) / scale
            target_mean = target.mean(0)
            centered = target - target_mean
            validation_x = (features["validation"][:, layer] - mean) / scale
            for ridge in lambdas:
                if ridge == 0:
                    coef = torch.linalg.lstsq(train_x, centered, driver="gelsd").solution
                else:
                    gram = train_x.T @ train_x
                    coef = torch.linalg.solve(gram + ridge * len(train_x) * torch.eye(gram.shape[0], dtype=gram.dtype), train_x.T @ centered)
                val_pred = validation_x @ coef + target_mean
                val_error = float((val_pred - banks["validation"]["z"].double()).square().mean())
                if math.isfinite(val_error) and (best is None or val_error < best[0]):
                    best = (val_error, layer, ridge, mean, scale, target_mean, coef)
        if best is None:
            raise ValueError("All probe fits were nonfinite")
        error, layer, ridge, mean, scale, target_mean, coef = best
        result: dict[str, Any] = {"selected_layer": layer, "selected_ridge": ridge, "validation_code_mse": error,
                                 "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
                                 "target_mean": target_mean.tolist(), "coefficients": coef.tolist(), "splits": {}}
        for key in ("validation", "id_test", "test"):
            batch = banks[key]
            code = (features[key][:, layer] - mean) / scale @ coef + target_mean
            truth = batch["z"].double()
            code_error = (code - truth).square()
            variance = (truth - truth.mean(0)).square().mean(0)
            code_r2 = torch.where(variance > 0, 1 - code_error.mean(0) / variance.clamp_min(1e-12), torch.full_like(variance, float("nan")))
            executed, valid = self.teacher(code, batch["x"][:, -1].double())
            execution_error = (executed - batch["y"].double()).square()
            denom = (batch["context_y"].double().mean(-1) - batch["y"].double()).square().mean()
            group = {}
            for pid in torch.unique(batch["pair_id"]).tolist():
                keep = batch["pair_id"] == pid
                group[str(pid)] = {"pair": self.pairs[pid].tolist(), "count": int(keep.sum()),
                                   "code_mse": float(code_error[keep].mean()),
                                   "executor_mse": float(execution_error[keep].mean()) if valid[keep].all() else None,
                                   "invalid_executor_count": int((~valid[keep]).sum())}
            # An invalid code invalidates the aggregate executor score, rather than
            # silently excluding a hard example and inflating reported performance.
            executor_mse = float(execution_error.mean()) if valid.all() else None
            result["splits"][key] = {"code_mse": float(code_error.mean()),
                "code_r2_by_coordinate": [float(v) if torch.isfinite(v) else None for v in code_r2],
                "code_r2_macro": float(code_r2[torch.isfinite(code_r2)].mean()) if torch.isfinite(code_r2).any() else None,
                "executor_mse": executor_mse,
                "executor_context_mean_r2": 1 - executor_mse / float(denom) if executor_mse is not None and denom > 0 else None,
                "invalid_executor_count": int((~valid).sum()), "count": len(code), "by_pair": group,
                "decoded_code": code.tolist(), "executor_prediction": [float(v) if torch.isfinite(v) else None for v in executed]}
        return result
