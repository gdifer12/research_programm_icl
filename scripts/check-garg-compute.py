#!/usr/bin/env python3
"""Bounded equivalence check for benchmark-only Garg compute variants."""
import argparse
import copy
import sys
from datetime import datetime, timezone

import torch

from icl_lab import config as configuration, engine
from icl_lab.runtime import ROOT, project_path, require_local_environment
from icl_lab.storage import sha, write_json
from icl_lab.tasks import make_task

sys.path.insert(0, str(ROOT / 'scripts'))
from garg_benchmark_compute import configure_compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_local_environment()
    config = configuration.load(ROOT / 'configs/experiments/garg.yaml', seed=1)
    config['model'].update(n_embd=32, n_layer=2, n_head=4)
    device = engine.setup(config, args.device)
    task = make_task(config)
    initial = task.build_model()
    generator = torch.Generator().manual_seed(9031)
    rows = []
    for sample_step in [0, 16000, 30000]:
        batch = task.sample(4, generator, 'train', sample_step)
        reference = None
        for mode in ['baseline', 'cached_mask', 'cached_mask_norm']:
            model = copy.deepcopy(initial).to(device)
            keys = set(model.state_dict())
            cpu_rng = torch.get_rng_state().clone()
            mps_rng = torch.mps.get_rng_state().clone() if device.type == 'mps' else None
            norm_fn = configure_compute(model, mode)
            assert set(model.state_dict()) == keys
            x = batch['x'].to(device).detach().requires_grad_(True)
            y = batch['y'].to(device)
            pred = model(x, y)
            loss = task.loss(pred, {'y': y})
            parameters = list(model.parameters())
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
            query = torch.autograd.grad(pred[:, -1].sum(), x)[0]
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient.detach().clone()
            norm = norm_fn(parameters)
            assert torch.isfinite(norm)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            if mps_rng is not None:
                assert torch.equal(mps_rng, torch.mps.get_rng_state())
            values = {'output': pred.detach().cpu(), 'loss': loss.detach().cpu(),
                      'query_gradient': query.detach().cpu(), 'norm': norm.detach().cpu(),
                      'parameter_gradients': torch.cat([g.detach().cpu().flatten() for g in gradients])}
            if reference is None:
                reference = values
            errors = {key: float((value - reference[key]).abs().max()) for key, value in values.items()}
            for key, value in values.items():
                torch.testing.assert_close(value, reference[key], rtol=2e-5, atol=2e-6, msg=f'{mode}: {key}')
            rows.append({'mode': mode, 'pairs': x.shape[1], 'max_abs_difference': errors})
    result = {'state': 'passed', 'device': str(device), 'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Reduced-model forward/parameter-gradient/query-gradient equivalence, not convergence',
              'model': config['model'], 'batch': 4, 'rtol': 2e-5, 'atol': 2e-6, 'cases': rows,
              'rng_and_state_dict_preserved': True, 'helper_sha256': sha(ROOT / 'scripts/garg_benchmark_compute.py')}
    write_json(project_path(args.output), result)
    print('Equivalence passed:', args.device, 'pairs 11/27/41;', len(rows), 'cases')


if __name__ == '__main__':
    main()
