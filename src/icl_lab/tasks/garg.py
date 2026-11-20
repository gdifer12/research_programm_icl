"""Gaussian linear regression; fresh nanoGPT-style model, no author-weight compatibility."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads, layers):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = [z.reshape(b, t, self.heads, w // self.heads).transpose(1, 2)
                   for z in self.qkv(self.norm1(x)).chunk(3, dim=-1)]
        # Explicit attention supports higher-order query gradients on CPU and V100 alike.
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(w // self.heads)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        attention = scores.masked_fill(~mask, float('-inf')).softmax(-1)
        x = x + self.proj((attention @ v).transpose(1, 2).reshape(b, t, w))
        return x + self.mlp(self.norm2(x))


class RegressionTransformer(nn.Module):
    def __init__(self, dims, pairs, config):
        super().__init__()
        width, layers, heads = config['n_embd'], config['n_layer'], config['n_head']
        if width % heads:
            raise ValueError('n_embd must be divisible by n_head')
        self.read_in = nn.Linear(dims, width)
        self.positions = nn.Embedding(2 * pairs, width)
        self.blocks = nn.ModuleList([Block(width, heads, layers) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.read_out = nn.Linear(width, 1)
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(block.mlp[-1].weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y, features=False):
        y_tokens = F.pad(y.unsqueeze(-1), (0, x.shape[-1] - 1))
        tokens = torch.stack([x, y_tokens], dim=2).reshape(x.shape[0], -1, x.shape[-1])
        h = self.read_in(tokens) + self.positions(torch.arange(tokens.shape[1], device=x.device))
        saved = {}
        for i, block in enumerate(self.blocks):
            h = block(h)
            if features:
                saved[f'layer_{i}'] = h[:, ::2]
        h = self.norm(h)
        pred = self.read_out(h[:, ::2]).squeeze(-1)
        return (pred, saved) if features else pred


class Task:
    def __init__(self, config):
        self.config = config
        self.dims = config['data']['n_dims']
        self.pairs = config['data']['n_pairs']

    def state_dict(self):
        return {'n_dims': self.dims, 'n_pairs': self.pairs}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError('Task-state mismatch')

    def sample(self, batch_size, generator, split='train', step=0):
        dims, pairs = self.dims, self.pairs
        curriculum = self.config['data'].get('curriculum', {})
        if split == 'train' and curriculum.get('enabled', False):
            interval = curriculum['interval']
            dims = min(dims, curriculum['dims_start'] + step // interval)
            pairs = min(pairs, curriculum['pairs_start'] + 2 * (step // interval))
        x = torch.randn(batch_size, pairs, self.dims, generator=generator)
        x[:, :, dims:] = 0
        w = torch.randn(batch_size, self.dims, generator=generator)
        y = (x * w[:, None]).sum(-1)
        return {'x': x, 'y': y, 'w': w, 'active_dims': torch.full((batch_size,), dims),
                'episode_id': torch.arange(batch_size)}

    def build_model(self):
        return RegressionTransformer(self.dims, self.pairs, self.config['model'])

    def predict(self, model, batch):
        return model(batch['x'], batch['y'])

    def loss(self, pred, batch):
        return F.mse_loss(pred, batch['y'])

    def metrics(self, pred, batch):
        error = (pred - batch['y']).square()
        return {'mse': error, 'nmse': error / self.dims}

    def baseline_predictions(self, batch):
        x, y = batch['x'].cpu().double(), batch['y'].cpu().double()
        outputs = {k: torch.zeros_like(y) for k in ['least_squares', 'three_nn', 'averaging', 'zero']}
        for k in range(1, x.shape[1]):
            a, b, query = x[:, :k], y[:, :k], x[:, k]
            solution = torch.linalg.lstsq(a, b.unsqueeze(-1), driver='gelsd').solution.squeeze(-1)
            outputs['least_squares'][:, k] = (solution * query).sum(-1)
            weight = (a * b.unsqueeze(-1)).mean(1)
            outputs['averaging'][:, k] = (weight * query).sum(-1)
            closest = (a - query[:, None]).square().sum(-1).topk(min(3, k), largest=False).indices
            outputs['three_nn'][:, k] = b.gather(1, closest).mean(-1)
        return {key: value.float() for key, value in outputs.items()}

    def activations(self, model, batch):
        return model(batch['x'], batch['y'], features=True)[1]

    def diagnostics(self, model, banks, device):
        bank = banks['test']
        x = bank['x'].to(device).detach().requires_grad_(True)
        y = bank['y'].to(device)
        ks = sorted({min(self.dims // 2, self.pairs - 1), min(self.dims, self.pairs - 1), self.pairs - 1})
        records = []
        # Differentiation holds the supplied contextual labels fixed.
        with torch.enable_grad():
            pred = model(x, y)
            for k in ks:
                grad = torch.autograd.grad(pred[:, k].sum(), x, retain_graph=True)[0][:, k]
                a, b = x[:, :k].detach().cpu().double(), y[:, :k].cpu().double()
                reference = torch.linalg.lstsq(a, b.unsqueeze(-1), driver='gelsd').solution.squeeze(-1).float().to(device)
                records.append({'k': k, 'gradient_mse_vs_least_squares': (grad-reference).square().mean().item(),
                                'gradient_cosine_vs_least_squares': F.cosine_similarity(grad, reference).mean().item()})
        return {'query_gradient': records,
                'interpretation': 'Functional local correctness; does not identify the internal algorithm.'}
