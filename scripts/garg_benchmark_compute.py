"""Benchmark-only Garg compute variants; production modules are untouched."""
import math
from types import MethodType

from icl_lab.runtime import require_local_environment

require_local_environment()

import torch


def _cached_forward(self, x):
    b, t, w = x.shape
    q, k, v = [z.reshape(b, t, self.heads, w // self.heads).transpose(1, 2)
               for z in self.qkv(self.norm1(x)).chunk(3, dim=-1)]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(w // self.heads)
    mask = self._benchmark_causal_mask[:t, :t]
    attention = scores.masked_fill(~mask, float('-inf')).softmax(-1)
    x = x + self.proj((attention @ v).transpose(1, 2).reshape(b, t, w))
    return x + self.mlp(self.norm2(x))


def _baseline_norm(parameters):
    return torch.nn.utils.clip_grad_norm_(parameters, float('inf'))


def _norm_without_scaling(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    return torch.nn.utils.get_total_norm(gradients, norm_type=2.0, foreach=False)


def configure_compute(model, mode):
    """Configure a benchmark model and return ``norm_fn(model.parameters())``."""
    if mode not in {'baseline', 'cached_mask', 'cached_mask_norm'}:
        raise ValueError(f'Unknown Garg benchmark compute mode: {mode}')
    for block in model.blocks:
        if '_benchmark_causal_mask' in block._buffers:
            delattr(block, '_benchmark_causal_mask')
        if getattr(block.forward, '__func__', None) is _cached_forward:
            delattr(block, 'forward')
        if mode != 'baseline':
            length = model.positions.num_embeddings
            mask = torch.ones(length, length, device=block.qkv.weight.device,
                              dtype=torch.bool).tril()
            block.register_buffer('_benchmark_causal_mask', mask, persistent=False)
            block.forward = MethodType(_cached_forward, block)
    return _norm_without_scaling if mode == 'cached_mask_norm' else _baseline_norm
