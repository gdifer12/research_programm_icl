from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml
from .runtime import project_path

DEFAULTS = {
    'schema_version': 1, 'task_seed': 17, 'model_seed': 1, 'train_seed': 2, 'eval_seed': 3,
    'train': {'steps': 1000, 'batch_size': 32, 'lr': 1e-3, 'optimizer': 'adamw',
              'weight_decay': 0.0, 'betas': [0.9, 0.999], 'scheduler': 'constant',
              'warmup_steps': 0, 'grad_clip': 1.0, 'eval_every': 100, 'save_every': 100,
              'log_every': 10, 'precision': 'fp32', 'deterministic': True, 'threads': 4,
              'analysis_checkpoints': 20, 'analysis_every': 0},
    'eval': {'episodes': 1280, 'batch_size': 128, 'diagnostic_episodes': 256,
             'bootstrap_samples': 1000},
    'analysis': {}, 'data': {}, 'model': {},
}


def merge(base, extra):
    out = deepcopy(base)
    for key, value in extra.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else deepcopy(value)
    return out


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(path, smoke=False, steps=None, seed=None):
    raw = yaml.safe_load(project_path(path, 'config', must_exist=True).read_text())
    config = merge(DEFAULTS, raw)
    overrides = config.pop('smoke', {})
    if smoke:
        config = merge(config, {'train': {'steps': 4, 'batch_size': 4, 'eval_every': 2,
                                         'save_every': 2, 'log_every': 1, 'threads': 1,
                                         'analysis_checkpoints': 3},
                                'eval': {'episodes': 16, 'batch_size': 8,
                                         'diagnostic_episodes': 16, 'bootstrap_samples': 50}})
        config = merge(config, overrides)
    config['technical_smoke'] = smoke
    if steps is not None:
        config['train']['steps'] = steps
    if seed is not None:
        config['model_seed'] = seed
        config['train_seed'] = seed + 10000
    validate(config)
    return config


def validate(c):
    if c.get('schema_version') != 1 or c.get('experiment') not in {'garg', 'assign_add', 'kobayashi'}:
        raise ValueError('Expected schema_version=1 and a supported experiment')
    for section, names in [('train', ['steps', 'batch_size', 'eval_every', 'save_every', 'log_every', 'threads']),
                           ('eval', ['episodes', 'batch_size', 'diagnostic_episodes'])]:
        for name in names:
            if not isinstance(c[section][name], int) or c[section][name] <= 0:
                raise ValueError(f'{section}.{name} must be a positive integer')
    if c['train']['precision'] not in {'fp32', 'fp16'}:
        raise ValueError('Use fp32 or fp16; V100 does not support native bf16')
    if c['train']['optimizer'].lower() not in {'adam', 'adamw'}:
        raise ValueError('Supported optimizers: adam, adamw')
    if c['train']['scheduler'] not in {'constant', 'cosine'}:
        raise ValueError('Supported schedules: constant, cosine')
    if not 0 <= c['train']['warmup_steps'] < c['train']['steps']:
        raise ValueError('warmup_steps must be smaller than steps')
    if c['train']['lr'] <= 0 or c['train']['weight_decay'] < 0:
        raise ValueError('Invalid optimizer configuration')
    for name, minimum in [('analysis_checkpoints', 1), ('analysis_every', 0)]:
        if not isinstance(c['train'][name], int) or c['train'][name] < minimum:
            raise ValueError(f'train.{name} must be an integer >= {minimum}')
