#!/usr/bin/env python3
"""Bounded, benchmark-only Garg compute A/B; production code remains unchanged."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gc
import importlib.util
from pathlib import Path

from icl_lab.runtime import ROOT, project_path, require_local_environment
import torch


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, project_path(ROOT / 'scripts' / filename, must_exist=True))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = load_script('garg_throughput_timing', 'benchmark-garg-throughput.py')
compute = load_script('garg_compute_variants', 'garg_benchmark_compute.py')
MODES = ['baseline', 'cached_mask', 'cached_mask_norm']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON or directory inside artifacts/validation/garg-throughput-20261119')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--timed', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if not (0 <= args.warmup <= 200 and 1 <= args.timed <= 1000 and 1 <= args.repeats <= 6):
        parser.error('Require warmup 0..200, timed 1..1000, repeats 1..6')
    args.sample_step = 30000
    require_local_environment()
    base = project_path(timing.BASE, 'benchmark directory')
    base.mkdir(parents=True, exist_ok=True)
    destination = args.output or base / ('compute-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    destination = project_path(destination if destination.is_absolute() else ROOT / destination, 'benchmark output')
    result_path = project_path(destination if destination.suffix == '.json' else destination / 'result.json')
    if not result_path.is_relative_to(base) or result_path.exists():
        raise ValueError('Output must be a fresh JSON inside ' + str(base))
    original_build = timing.build
    current_norm = None
    active_mode = None

    def build(config):
        nonlocal current_norm
        values = original_build(config)
        current_norm = compute.configure_compute(values[2], active_mode)
        return values

    def update(next_batch, task, model, optimizer, scaler, device, parts=None):
        if parts is not None:
            raise ValueError('Component profiling is outside this A/B benchmark')
        batch = timing.engine.move(next_batch(), device)
        optimizer.zero_grad(set_to_none=True)
        prediction = task.predict(model, batch)
        loss = task.loss(prediction, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite benchmark loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = current_norm(model.parameters())
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite benchmark gradient')
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    timing.build, timing.update = build, update
    with ExitStack() as stack:
        for path in [ROOT / 'artifacts/runs/kobayashi-jupyter-20261119/.gpu-series.lock', base / '.benchmark.lock']:
            path = project_path(path, 'GPU lock')
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open('a+'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('GPU/benchmark lock busy: ' + str(path)) from None
        config = timing.configuration.load(project_path(ROOT / 'configs/experiments/garg.yaml', must_exist=True), seed=1)
        train = config['train']
        if (config['experiment'] != 'garg' or train['batch_size'] != 64 or train['precision'] != 'fp32'
                or not train['deterministic'] or train['threads'] != 4 or train['grad_clip'] != 0
                or train['optimizer'] != 'adam' or train['scheduler'] != 'constant' or train['warmup_steps']):
            raise ValueError('Expected unchanged full Garg config, constant Adam LR, FP32, batch64 and no gradient clipping')
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = {'purpose': 'technical_compute_timing_only; not a scientific experiment', 'state': 'running',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'config_resolved': config,
                  'config_hash': timing.configuration.digest(config), 'code_hash': timing.code_hash(),
                  'script_sha256': timing.sha(project_path(__file__)),
                  'throughput_script_sha256': timing.sha(project_path(timing.__file__)),
                  'compute_helper_sha256': timing.sha(project_path(compute.__file__)),
                  'sample_step': args.sample_step, 'warmup_updates': args.warmup,
                  'timed_updates': args.timed, 'repeats': args.repeats, 'cases': [],
                  'limitations': ['Benchmark-only model instances; no production changes or checkpoints.',
                                  'CPU data generation remains synchronous in every compute mode.',
                                  'Reduced-model MPS numerical equivalence is a separate prerequisite, not performed by this runner.',
                                  'Short counterbalanced timing does not establish long-run thermal performance.',
                                  'MPS numerical nondeterminism can change losses despite identical initial weights and input hashes.']}
        timing.write_json(result_path, result)
        try:
            result['environment'] = timing.environment(timing.engine.setup(config, 'mps'))
            for repeat in range(args.repeats):
                for active_mode in MODES if repeat % 2 == 0 else reversed(MODES):
                    print(f'compute repeat {repeat + 1}/{args.repeats}: {active_mode}', flush=True)
                    case = timing.run_case(config, args, 'baseline')
                    result['cases'].append({**case, 'compute_mode': active_mode, 'repeat': repeat + 1,
                                            'order': len(result['cases']) + 1})
                    timing.write_json(result_path, result)
                    gc.collect()
                    torch.mps.empty_cache()
            for key in ['initial_weights_sha256', 'first_timed_batch_sha256', 'last_timed_batch_sha256', 'generator_after_sha256']:
                if len({case[key] for case in result['cases']}) != 1:
                    raise RuntimeError('Benchmark cases differ: ' + key)
            result['state'] = 'completed_timing_only'
        except BaseException as error:
            result.update(state='failed', error=repr(error))
            raise
        finally:
            result['finished_at'] = datetime.now(timezone.utc).isoformat()
            timing.write_json(result_path, result)
            gc.collect()
        print(result_path, flush=True)


if __name__ == '__main__':
    main()
