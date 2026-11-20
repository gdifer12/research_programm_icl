"""Notebook interface to the project-local Assign and Add generator and model."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil

from icl_lab.runtime import ROOT, project_path
from notebook_python import inspect_json


def _science_process(action, run_dir, rows=None):
    arguments = [action, '--run-dir', str(project_path(run_dir, 'training run', True))]
    if rows is not None:
        if len(rows) == 0:
            raise ValueError('Выберите хотя бы одну строку банка.')
        arguments.extend(['--rows', *map(str, rows)])
    return inspect_json(Path(__file__), arguments)


def preview_data(run_dir, rows=(4, 8, 18)):
    """Generate a train example and decode selected rows of this run's OOD bank."""
    return _science_process('preview', run_dir, rows)


def predict_examples(run_dir, rows=(4, 8, 18)):
    """Load final weights and predict the selected OOD rows on CPU."""
    return _science_process('predict', run_dir, rows)


def launch_new_series(name):
    """Reserve a fresh series directory and start the existing three-seed worker."""
    if not name or name in {'.', '..'} or Path(name).name != name:
        raise ValueError('Укажите имя нового каталога внутри artifacts/runs/.')
    series = project_path(ROOT / 'artifacts/runs' / name, 'new series')
    command = [str(ROOT / 'scripts/local-python'),
               str(ROOT / 'scripts/run-assign-series.py'), '--run-root', str(series),
               '--seeds', '1', '2', '3']
    if caffeinate := shutil.which('caffeinate'):
        command = [caffeinate, '-i', *command]
    # An atomic reservation also protects against repeating the launch cell.
    series.mkdir(parents=True, exist_ok=False)
    with (series / 'worker.log').open('x') as log:
        subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    return series


def _decode(task, batch, row):
    tokens = batch['tokens'][row].tolist()
    prefix = [token for token in tokens[:-4] if token != task.pad]
    names = {task.first_var + i: chr(ord('a') + i) for i in task.vars}
    assignments = ', '.join(f'{names[key]}={value}' for key, value in zip(prefix[::2], prefix[1::2]))
    left, right = [names.get(token, str(token)) for token in tokens[-3:-1]]
    a, b = batch['operands'][row].tolist()
    return {'Присваивания': assignments, 'Запрос': f'{left} + {right}',
            'Операнды': f'{a} + {b}', 'Ответ': batch['target'][row].item()}


def _examples(action, run, rows):
    # These imports belong to the scientific environment. The notebook kernel
    # only calls the public functions above and never needs to import torch.
    import torch
    import yaml
    from icl_lab.engine import load_run, read_bank
    from icl_lab.tasks import make_task

    run = project_path(run, 'training run', True)
    config = yaml.safe_load(project_path(run / 'config.yaml', 'run config', True).read_text())
    if config['experiment'] != 'assign_add':
        raise ValueError('Нужен каталог запуска Assign and Add.')
    if action == 'circuit':
        _, task, model, _, _, _ = load_run(run, checkpoint='final', device='cpu')
        values = torch.arange(task.modulus)
        pairs = torch.cartesian_prod(values, values)
        with torch.inference_mode():
            predicted = model.isolated_addition(pairs).argmax(-1)
        correct = predicted == pairs.sum(dim=1) % task.modulus
        seen = task.pair_mask.flatten()
        return {'all_pairs': len(pairs), 'accuracy': correct.float().mean().item(),
                'seen_accuracy': correct[seen].float().mean().item(),
                'held_out_accuracy': correct[~seen].float().mean().item()}

    bank, _ = read_bank(run, 'test')
    if any(not 0 <= row < len(bank['tokens']) for row in rows):
        raise IndexError('Номер строки за пределами тестового банка.')
    if action == 'preview':
        task = make_task(config)
        sample = task.sample(1, torch.Generator().manual_seed(17), split='train')
        examples = [{'Источник': 'новый train', **_decode(task, sample, 0)}]
        for row in rows:
            examples.append({'Источник': f'test[{row}]', **_decode(task, bank, row)})
        return examples

    _, task, model, _, _, _ = load_run(run, checkpoint='final', device='cpu')
    with torch.inference_mode():
        logits = task.predict(model, {'tokens': bank['tokens'][rows]})
        predicted = logits.argmax(dim=-1).tolist()
    return [{'Строка test': row, 'Запрос': _decode(task, bank, row)['Запрос'],
             'Правильный ответ': bank['target'][row].item(), 'Предсказание': answer}
            for row, answer in zip(rows, predicted)]


def _active_artifacts(run, pattern):
    """Keep the latest evaluation for each active (split, step) after a resume."""
    files = sorted(run.glob(pattern), key=lambda path: (path.stat().st_mtime_ns, str(path)))
    if not files:
        return []
    events_path = project_path(run / 'events.jsonl', must_exist=True)
    events = (json.loads(line) for line in events_path.read_text().splitlines() if line.strip())
    starts = [event for event in events if event['event'] == 'start']
    active = {}
    for path in files:
        artifact = json.loads(project_path(path, must_exist=True).read_text())
        step = artifact['step']
        for i, start in enumerate(starts):
            if start['attempt_id'] == artifact['attempt_id']:
                if all(step <= later['resume_from_step'] for later in starts[i + 1:]):
                    active[artifact.get('split'), step] = (path, artifact)
                break
    return active.values()


def evaluation_history(series_dir, seeds):
    """Read checkpoint accuracy on the fixed ID and OOD banks of each training run."""
    import pandas as pd

    rows = []
    for seed in seeds:
        run = project_path(series_dir / f'seed-{seed}')
        for _, summary in _active_artifacts(run, 'evaluations/*/summary.json'):
            split = summary['split']
            if split not in ('id_test', 'test'):
                continue
            for metric, value in summary['model'].items():
                group = metric.removeprefix('accuracy/')
                if metric == 'accuracy':
                    group = 'all'
                elif not (metric.startswith('accuracy/') and ('_ood' in group or group.endswith('_id'))):
                    continue
                if value['count']:
                    rows.append((seed, summary['step'], split, group, value['mean'], value['count']))
    columns = ['seed', 'step', 'split', 'group', 'accuracy', 'n']
    return pd.DataFrame(rows, columns=columns).sort_values(columns[:4]).reset_index(drop=True)


def final_circuit_data(series_dir, seeds):
    """Apply the trained isolated addition circuit to every pair, on CPU."""
    import pandas as pd

    return pd.DataFrame([{'seed': seed, **_science_process('circuit', series_dir / f'seed-{seed}')}
                         for seed in seeds])


def check_final_scores(series_dir, seeds):
    """Score the saved predictions against labels from each fixed bank."""
    import numpy as np
    import pandas as pd
    import yaml

    rows = []
    common_banks = {}
    for seed in seeds:
        run = project_path(series_dir / f'seed-{seed}', must_exist=True)
        config = yaml.safe_load(project_path(run / 'config.yaml', must_exist=True).read_text())
        status = json.loads(project_path(run / 'status.json', must_exist=True).read_text())
        step = config['train']['steps']
        if (config['experiment'] != 'assign_add' or config['model_seed'] != seed
                or config['technical_smoke'] or status['state'] != 'completed' or status['step'] != step):
            raise ValueError(f'Сид {seed}: нужен завершённый запуск Assign and Add с полным бюджетом.')
        banks = json.loads(project_path(run / 'banks/manifest.json', must_exist=True).read_text())['banks']
        code_hash = json.loads(project_path(run / 'manifest.json', must_exist=True).read_text())['code_hash']
        for split in ('id_test', 'test'):
            info = banks[split]
            active = _active_artifacts(run, f'evaluations/{split}-{step:08d}-*/summary.json')
            summary_path, summary = next(((p, s) for p, s in active
                                          if s['split'] == split and s['step'] == step), (None, None))
            if (summary is None or summary['bank_sha256'] != info['sha256']
                    or summary['code_hash'] != code_hash or summary['technical_smoke']):
                raise ValueError(f'Сид {seed}: отсутствует подходящая финальная оценка {split}.')
            if common_banks.setdefault(split, info['sha256']) != info['sha256']:
                raise ValueError(f'Для разных сидов использованы разные банки {split}.')
            with np.load(project_path(run / 'banks' / info['file'], must_exist=True)) as bank:
                ids, targets = bank['episode_id'], bank['target']
            with np.load(project_path(summary_path.with_name('predictions.npz'), must_exist=True)) as predictions:
                if not np.array_equal(ids, predictions['episode_id']):
                    raise ValueError(f'Сид {seed}: порядок предсказаний отличается от банка {split}.')
                logits = predictions['prediction']
            if logits.shape != (len(targets), config['data']['modulus']) or not np.isfinite(logits).all():
                raise ValueError(f'Сид {seed}: неверная форма или значения логитов {split}.')
            correct = int((logits.argmax(axis=1) == targets).sum())
            metric = summary['model']['accuracy']
            if (len(targets) != info['episodes'] or len(targets) != metric['count']
                    or not np.isclose(correct / len(targets), metric['mean'])):
                raise ValueError(f'Сид {seed}: пересчёт точности расходится с оценкой {split}.')
            rows.append((seed, step, split, len(targets), correct / len(targets)))
    return pd.DataFrame(rows, columns=['seed', 'step', 'split', 'n', 'accuracy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preview', 'predict', 'circuit'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=(4, 8, 18))
    args = parser.parse_args()
    print(json.dumps(_examples(args.action, args.run_dir, args.rows), ensure_ascii=False))
