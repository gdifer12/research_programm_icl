#!/usr/bin/env python3
"""Extrapolate the current sequential suite from a bounded MPS timing record."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from icl_lab import config as configuration
from icl_lab.runtime import ROOT, project_path, require_local_environment
from icl_lab.storage import sha, write_json


LABELS = {'garg': 'Garg', 'assign_add': 'Assign and Add',
          'kobayashi': 'Kobayashi · transformer',
          'kobayashi_hyper': 'Kobayashi · hypernetwork'}


def main():
    require_local_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = project_path(args.input, 'timing summary', True)
    output = project_path(args.output, 'budget record')
    measured = json.loads(source.read_text())
    if measured['state'] != 'completed_timing_only':
        raise ValueError('The complete four-experiment timing record is required')
    suite_path = project_path(ROOT / 'configs/suites/three-experiments.yaml', must_exist=True)
    suite = yaml.safe_load(suite_path.read_text())
    rows = []
    for entry in suite['experiments']:
        config_path = project_path(ROOT / entry['config'], must_exist=True)
        key = config_path.stem
        config = configuration.load(config_path, seed=1)
        run = measured['runs'][key]
        measured_config = yaml.safe_load(project_path(source.parent / key / 'config.yaml', must_exist=True).read_text())
        if run['state'] != 'completed_timing_only':
            raise ValueError(f'Incomplete timing for {key}')
        for section in ['data', 'model', 'train', 'eval', 'analysis']:
            if config[section] != measured_config[section]:
                raise ValueError(f'{key}: current {section} differs from the measured full configuration')
        count = run['counts']
        steps, batch = config['train']['steps'], config['train']['batch_size']
        if count['training_updates'] != steps:
            raise ValueError(f'{key}: training budget differs')
        evaluation = sum(value['wall_seconds'] for value in run['evaluation'].values())
        parts = {
            'training': steps * run['updates']['mean_seconds'],
            'validation': count['validation_calls'] * run['validation']['mean_seconds'],
            'resume_checkpoints': count['full_checkpoint_saves_including_initial'] * run['full_save_and_previous_hash']['mean_seconds'],
            'model_checkpoints': count['model_only_checkpoint_saves_including_initial_final'] * run['model_only_save']['mean_seconds'],
            'bank_preparation': run['prepare_banks_wall_seconds'],
            'final_evaluation_both_splits': evaluation,
            'final_diagnostics': run['analysis_wall_seconds'],
            'report': run['report_wall_seconds'],
        }
        total = sum(parts.values())
        timings = {name: run[name] for name in ['updates', 'update_blocks', 'validation',
                   'model_only_save', 'full_save_and_previous_hash', 'evaluation']}
        rows.append({'key': key, 'label': LABELS[key], 'config': entry['config'],
                     'config_sha256': sha(config_path), 'code_hash': run['code_hash'],
                     'epochs': None, 'steps_per_seed': steps, 'batch_size': batch,
                     'training_episodes_per_seed': steps * batch, 'seeds': entry['seeds'],
                     'counts': count, 'measured_timings': timings,
                     'suite_components_seconds_per_seed': parts,
                     'suite_seconds_per_seed': total,
                     'suite_seconds_all_seeds': total * len(entry['seeds']),
                     'local_run_extra_seconds_per_seed': (count['sweep_checkpoints_including_initial'] - 1) * evaluation,
                     'analyze_all_extra_seconds_per_seed': (count['analyze_all_calls'] - 1) * run['analysis_wall_seconds']})
    hardware = measured['hardware']
    env = next(iter(measured['runs'].values()))['environment']
    record = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Planning estimate; not a completed scientific experiment',
        'machine_label': f"{hardware['machdep.cpu.brand_string']} · {int(hardware['hw.memsize']) / 2**30:.0f} ГБ · {env['platform']}",
        'hardware': hardware, 'environment': {k: env[k] for k in ['python', 'torch', 'platform', 'device', 'mps_cpu_fallback']},
        'source_summary': str(source.relative_to(ROOT)), 'source_summary_sha256': sha(source),
        'benchmark_script': 'scripts/benchmark-mps.py',
        'benchmark_script_sha256': sha(ROOT / 'scripts/benchmark-mps.py'),
        'suite': str(suite_path.relative_to(ROOT)), 'suite_sha256': sha(suite_path),
        'measurement_protocol': {'warmup_updates_per_configuration': 20,
                                 'timed_updates_per_configuration': 120,
                                 'full_architecture_and_batch': True, 'mps_synchronized': True,
                                 'garg_shape': 'mature curriculum; all 500001 updates priced at full shape'},
        'route': 'suite --execute: sequential training, periodic validation/saves, final ID/test evaluation, final analysis and report',
        'planning_margin': .30,
        'planning_margin_meaning': 'Discretionary time reserve, not a confidence interval or measured upper bound',
        'limitations': [
            '120 timed updates after warmup do not characterize hours of heating, sleep or competing workloads.',
            'Full mature shape overestimates the first 30000 Garg updates; no early-curriculum discount applied.',
            'Initial setup, provenance, logging, checkpoint rotation metadata and final suite aggregation omitted.',
            'Evaluation and diagnostics measured on early weights; report with a short log.',
            'The +30% reserve is a planning convention and does not guarantee completion within that time.',
            'All-checkpoint evaluation/analysis are separate additions; no long scientific run was executed.'
        ],
        'experiments': rows,
        'total_runs': sum(len(row['seeds']) for row in rows),
        'total_suite_seconds': sum(row['suite_seconds_all_seeds'] for row in rows),
    }
    write_json(output, record)
    print(json.dumps({'output': str(output), 'hours': record['total_suite_seconds'] / 3600,
                      'experiments': {row['key']: {'minutes_per_seed': row['suite_seconds_per_seed'] / 60,
                                     'hours_all_seeds': row['suite_seconds_all_seeds'] / 3600,
                                     'local_run_extra_minutes': row['local_run_extra_seconds_per_seed'] / 60}
                                     for row in rows}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
