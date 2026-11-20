#!/usr/bin/env python3
"""Bounded timing only: run with scripts/local-python scripts/benchmark-mps.py.

Uses full architectures/batches, but does not run a scientific training schedule.
Every invocation creates a fresh child of artifacts/validation/mps-budget-20261118.
"""
import gc
import math
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

from icl_lab import analysis, config as configuration, engine
from icl_lab.devices import memory_stats, synchronize
from icl_lab.runtime import ROOT, project_path, require_local_environment
from icl_lab.storage import (append_json, atomic_bytes, code_hash, environment,
                             save_torch, sha, write_json)
from icl_lab.tasks import make_task


WARMUP = 20
TIMED = 120
BLOCK = 30
BASE = ROOT / 'artifacts/validation/mps-budget-20261118'


def stats(values):
    return {'samples_seconds': values, 'mean_seconds': statistics.mean(values),
            'median_seconds': statistics.median(values), 'min_seconds': min(values),
            'max_seconds': max(values)}


def timed(call, device):
    synchronize(device)
    start = time.perf_counter()
    result = call()
    synchronize(device)
    return time.perf_counter() - start, result


def counts(config):
    train = config['train']
    steps = train['steps']
    selected = set(np.unique(np.geomspace(1, steps, train['analysis_checkpoints']).astype(int)))
    if train['analysis_every']:
        selected.update(range(train['analysis_every'], steps + 1, train['analysis_every']))
    selected.add(steps)
    return {'training_updates': steps,
            'validation_calls': math.ceil(steps / train['eval_every']),
            'validation_batches_per_call': math.ceil(config['eval']['episodes'] / config['eval']['batch_size']),
            'full_checkpoint_saves_including_initial': 1 + math.ceil(steps / train['save_every']),
            'model_only_checkpoint_saves_including_initial_final': len(selected) + 2,
            'sweep_checkpoints_including_initial': len(selected) + 1,
            'suite_evaluation_calls': 2, 'local_run_evaluation_calls': 2 * (len(selected) + 1),
            'default_analysis_calls': 1, 'analyze_all_calls': len(selected) + 1}


def benchmark(source, output):
    name = source.stem
    run = project_path(output / name, 'timing run')
    run.mkdir()
    config = configuration.load(source, seed=1)
    # Preserve the full config sizes while making every derived artifact technical.
    config['technical_smoke'] = True
    config['benchmark_purpose'] = 'timing_only; full sizes, bounded updates, mature input shape'
    train_cfg = config['train']
    if train_cfg['precision'] != 'fp32' or train_cfg['threads'] != 4 or not train_cfg['deterministic']:
        raise ValueError('Expected current deterministic FP32 / 4-thread configs')
    device = engine.setup(config, 'mps')
    task = make_task(config)
    model = task.build_model().to(device)
    optimizer_cls = torch.optim.AdamW if train_cfg['optimizer'].lower() == 'adamw' else torch.optim.Adam
    optimizer = optimizer_cls([p for p in model.parameters() if p.requires_grad],
                              lr=train_cfg['lr'], betas=tuple(train_cfg['betas']),
                              weight_decay=train_cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    generator = torch.Generator().manual_seed(config['train_seed'])
    attempt = 'timing-' + uuid.uuid4().hex[:12]
    digest = configuration.digest(config)
    source_hash = code_hash()
    result = {'purpose': 'timing_only', 'config': str(source.relative_to(ROOT)),
              'seed': 1, 'warmup_updates': WARMUP, 'timed_updates': TIMED,
              'training_sample_step': train_cfg['steps'] - 1,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'code_hash': source_hash, 'config_hash': digest, 'counts': counts(config),
              'environment': environment(device), 'state': 'running'}
    result_path = project_path(run / 'timing.json', 'timing result')
    atomic_bytes(run / 'config.yaml', yaml.safe_dump(config, sort_keys=False).encode())
    write_json(run / 'manifest.json', result)
    append_json(run / 'events.jsonl', {'event': 'start', 'attempt_id': attempt,
                                      'resume_from_step': 0, 'purpose': 'timing_only'})
    write_json(run / 'status.json', {'state': 'timing_only', 'step': 0,
                                    'attempt_id': attempt, 'technical_smoke': True})

    def progress(stage):
        result['stage'] = stage
        write_json(result_path, result)
        print(f'{name}: {stage}', flush=True)

    try:
        progress('prepare full evaluation banks')
        before_banks = engine.rng_state(generator, device)
        result['prepare_banks_wall_seconds'], _ = timed(lambda: engine.prepare(config, run / 'banks', task), device)
        engine.restore_rng(before_banks, generator, device)
        val, _ = engine.read_bank(run, 'validation')

        progress('20 warmup + 120 timed updates')
        durations, warmup = [], []
        for index in range(WARMUP + TIMED):
            model.train()
            fraction = index / max(train_cfg['steps'] - 1, 1)
            factor = 0.5 * (1 + math.cos(math.pi * fraction)) if train_cfg['scheduler'] == 'cosine' else 1.0
            if index < train_cfg['warmup_steps']:
                factor = (index + 1) / train_cfg['warmup_steps']
            for group in optimizer.param_groups:
                group['lr'] = train_cfg['lr'] * factor
            synchronize(device)
            start = time.perf_counter()
            batch = engine.move(task.sample(train_cfg['batch_size'], generator, 'train', train_cfg['steps'] - 1), device)
            optimizer.zero_grad(set_to_none=True)
            pred = task.predict(model, batch)
            loss = task.loss(pred, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite benchmark loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'] or float('inf'))
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite benchmark gradient')
            scaler.step(optimizer)
            scaler.update()
            synchronize(device)
            duration = time.perf_counter() - start
            (warmup if index < WARMUP else durations).append(duration)
            if index >= WARMUP and (index + 1 - WARMUP) % BLOCK == 0:
                append_json(run / 'metrics.jsonl', {'step': index + 1, 'attempt_id': attempt,
                    'train_loss': loss.item(), 'lr': optimizer.param_groups[0]['lr'],
                    'grad_norm': norm.item(), 'update_seconds': duration, 'device': str(device)})
                print(f'{name}: timed updates {index + 1 - WARMUP}/{TIMED}', flush=True)
        result['warmup'] = stats(warmup)
        result['updates'] = stats(durations)
        result['update_blocks'] = [stats(durations[i:i + BLOCK]) for i in range(0, TIMED, BLOCK)]
        result['memory_after_updates'] = memory_stats(device)
        result['last_loss'] = loss.item()
        del batch, pred, loss, norm

        progress('validation ×3')
        validation_times = []
        summary = None
        for _ in range(3):
            def validation():
                _, measures, _, _ = engine.evaluate_model(task, model, val, config['eval']['batch_size'], device)
                return engine.summarize(measures)
            seconds, summary = timed(validation, device)
            validation_times.append(seconds)
        result['validation'] = stats(validation_times)
        append_json(run / 'validation.jsonl', {'step': WARMUP + TIMED, 'attempt_id': attempt,
                                              'metrics': summary, 'seconds': validation_times[-1]})

        progress('model-only save ×2; full save + prior hash ×3')
        checkpoint_dir = project_path(run / 'checkpoints', 'timing checkpoint directory')
        checkpoint_dir.mkdir()
        checkpoint = project_path(checkpoint_dir / 'timing.pt', 'timing checkpoint')

        def payload(full):
            item = {'schema_version': 1, 'step': WARMUP + TIMED, 'model': model.state_dict(),
                    'task_state': task.state_dict(), 'config_hash': digest, 'code_hash': source_hash,
                    'resumable': False, 'attempt_id': attempt, 'training_device': str(device),
                    'purpose': 'timing_only; not a scientific training checkpoint'}
            if full:
                item.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            rng=engine.rng_state(generator, device), attempted=WARMUP + TIMED, skipped=0)
            return item

        weight_times = [timed(lambda: save_torch(checkpoint, payload(False)), device)[0] for _ in range(2)]
        full_path = project_path(checkpoint_dir / 'timing-full.pt', 'full timing checkpoint')
        save_torch(full_path, payload(True))

        def save_full():
            sha(full_path)
            save_torch(full_path, payload(True))

        full_times = [timed(save_full, device)[0] for _ in range(3)]
        result['model_only_save'] = stats(weight_times)
        result['full_save_and_previous_hash'] = stats(full_times)
        result['checkpoint_bytes'] = {'model_only': checkpoint.stat().st_size, 'full': full_path.stat().st_size}
        result['checkpoint_timing_exclusion'] = 'full-save rotation metadata operations; checksum, fsync and old-file hash included'
        del model, optimizer, scaler, task, generator, val
        gc.collect()
        torch.mps.empty_cache()

        progress('evaluate ID and OOD, including baselines and output IO')
        result['evaluation'] = {}
        for split in ['id_test', 'test']:
            seconds, evaluation = timed(lambda: engine.evaluate(run, str(checkpoint), split, 'mps'), device)
            result['evaluation'][split] = {'wall_seconds': seconds, 'engine_elapsed_seconds': evaluation['elapsed_seconds']}
            write_json(result_path, result)
        progress('one full diagnostic analysis')
        result['analysis_wall_seconds'], _ = timed(lambda: analysis.analyze(run, str(checkpoint), 'mps'), device)
        write_json(run / 'status.json', {'state': 'timing_only', 'step': WARMUP + TIMED,
                                        'attempt_id': attempt, 'technical_smoke': True})
        progress('report')
        result['report_wall_seconds'], _ = timed(lambda: analysis.report(run), device)
        result['state'] = 'completed_timing_only'
        result['limitations'] = [
            'Short warmed run cannot measure prolonged thermal throttling or concurrent workloads.',
            'Garg uses the mature curriculum shape throughout; this overestimates its first 30000 updates.',
            'Evaluation/diagnostics use early trained weights, not converged scientific checkpoints.',
            'Fresh process startup, per-update logging, initial model setup and provenance snapshot are not timed.',
            'Analysis is measured once; all-checkpoint extrapolation assumes similar cost for other weights.',
            'MPS memory counters are current allocations, not measured peaks.'
        ]
        progress('completed timing only')
        return result
    except BaseException as error:
        result.update(state='failed', error=repr(error))
        write_json(result_path, result)
        raise
    finally:
        gc.collect()
        torch.mps.empty_cache()


def main():
    require_local_environment()
    suite_path = project_path(ROOT / 'configs/suites/three-experiments.yaml', must_exist=True)
    suite = yaml.safe_load(suite_path.read_text())
    configs = [project_path(ROOT / entry['config'], 'benchmark config', True) for entry in suite['experiments']]
    if len(configs) != 4:
        raise ValueError('Expected the four configurations from the current suite')
    base = project_path(BASE, 'timing artifact directory')
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    output = project_path(base / stamp, 'fresh timing invocation')
    output.mkdir()
    hardware = {}
    for key in ['machdep.cpu.brand_string', 'hw.memsize', 'hw.logicalcpu']:
        completed = subprocess.run(['/usr/sbin/sysctl', '-n', key], capture_output=True, text=True, timeout=10)
        hardware[key] = completed.stdout.strip() if completed.returncode == 0 else None
    summary = {'purpose': 'bounded full-shape MPS timing, not scientific results',
               'created_utc': stamp, 'hardware': hardware, 'runs': {}, 'state': 'running'}
    summary_path = project_path(output / 'summary.json', 'benchmark summary')
    write_json(summary_path, summary)
    print(f'Timing artifacts: {output}', flush=True)
    started = time.perf_counter()
    try:
        for source in configs:
            summary['runs'][source.stem] = benchmark(source, output)
            write_json(summary_path, summary)
        summary['state'] = 'completed_timing_only'
    except BaseException as error:
        summary.update(state='failed', error=repr(error))
        raise
    finally:
        summary['wall_seconds'] = time.perf_counter() - started
        write_json(summary_path, summary)


if __name__ == '__main__':
    main()
