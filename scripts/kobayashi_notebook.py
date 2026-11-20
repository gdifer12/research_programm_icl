"""Load a Kobayashi episode on CPU and score the models from saved experiment artifacts."""
import argparse
import json
from pathlib import Path

from icl_lab.runtime import ROOT, project_path
from notebook_python import inspect_json


def inspect_episode(series_dir, *, seed=1, split="test", row=0):
    """Return the full episode, teacher label and two live model predictions."""
    arguments = ["--series", str(project_path(series_dir, "Kobayashi series", True)),
                 "--seed", str(seed), "--split", split, "--row", str(row)]
    return inspect_json(Path(__file__), arguments)


def _inspect(series, seed, split, row):
    import torch
    from icl_lab.engine import load_run, read_bank

    result = {}
    predictions = {}
    bank_hash = None
    for variant in ("vanilla", "hypernetwork"):
        run = project_path(series / variant / f"seed-{seed}", "Kobayashi run", True)
        config, task, model, _, _, _ = load_run(run, checkpoint="final", device="cpu")
        if config["experiment"] != "kobayashi" or model.variant != variant:
            raise ValueError(f"Unexpected model at {run}.")
        bank, digest = read_bank(run, split)
        if bank_hash is not None and bank_hash != digest:
            raise ValueError(f"The two models have different {split} banks.")
        bank_hash = digest
        if not 0 <= row < len(bank["y"]):
            raise IndexError(f"Choose a row from 0 to {len(bank['y']) - 1}.")
        episode = {key: tensor[row:row + 1] for key, tensor in bank.items()}
        x, context_y, z = episode["x"], episode["context_y"], episode["z"]
        with torch.inference_mode():
            teacher_y, valid = task.teacher(z, x)
            if not valid.item():
                raise ValueError("Invalid teacher matrix for the selected episode.")
            labels = torch.cat((context_y, episode["y"][:, None]), dim=1)
            torch.testing.assert_close(teacher_y, labels, rtol=1e-5, atol=1e-6)
            predictions[variant] = task.predict(model, {"x": x, "context_y": context_y}).item()
        if not result:
            result = {"episode_id": episode["episode_id"].item(),
                      "pair": task.pairs[episode["pair_id"].item()].tolist(),
                      "z": z[0].tolist(), "x": x[0].tolist(), "context_y": context_y[0].tolist(),
                      "y": episode["y"].item()}
    return {**result, "predictions": predictions}


def _checkpoint_scores(run, config, split, bank_info, code_hash, cutoffs):
    import numpy as np
    from garg_notebook_io import read_json, sha256_file
    from icl_lab.config import digest

    info = bank_info['banks'][split]
    bank_path = project_path(run / 'banks' / info['file'], must_exist=True)
    if sha256_file(bank_path) != info['sha256']:
        raise ValueError(f'Изменился оценочный банк {bank_path}')
    with np.load(bank_path) as bank:
        ids = bank['episode_id']
        y = bank['y'].astype('float64')
        context_mean = bank['context_y'].mean(axis=1, dtype='float64')
    n = len(ids)
    if y.shape != (n,) or n != info['episodes'] or n != len(np.unique(ids)):
        raise ValueError(f'Некорректные эпизоды в {bank_path}')
    baseline_mse = np.mean((y - context_mean) ** 2)
    if not baseline_mse > 0:
        raise ValueError(f'Нулевой или нечисловой знаменатель R² в {bank_path}')

    rows = []
    ckpt_dir = project_path(run / 'checkpoints', must_exist=True)
    for checkpoint in [ckpt_dir / 'initial.pt', *sorted(ckpt_dir.glob('step-*.pt'))]:
        step = 0 if checkpoint.stem == 'initial' else int(checkpoint.stem.split('-')[1])
        checkpoint_sha = sha256_file(checkpoint)
        # evaluate() keys each output by weights, code, bank, evaluation settings and device.
        key = digest({'checkpoint': checkpoint_sha, 'bank': info['sha256'], 'code': code_hash,
                      'eval': config['eval'], 'device_type': 'mps', 'precision': 'fp32'})[:20]
        directory = run / 'evaluations' / f'{split}-{step:08d}-{key}'
        summary = read_json(directory / 'summary.json')
        if summary is None:
            continue
        if (summary['split'] != split or summary['step'] != step or summary['technical_smoke']
                or summary['checkpoint_sha256'] != checkpoint_sha or summary['code_hash'] != code_hash
                or summary['bank_sha256'] != info['sha256']):
            raise ValueError(f'Оценка относится к другим условиям: {directory}')
        if step > cutoffs.get(summary['attempt_id'], -1):
            continue
        with np.load(project_path(directory / 'predictions.npz', must_exist=True)) as predictions:
            prediction = predictions['prediction'].astype('float64')
            if prediction.shape != y.shape or not np.array_equal(ids, predictions['episode_id']):
                raise ValueError(f'Предсказания и эпизоды не совпадают: {directory}')
        mse = np.mean((prediction - y) ** 2)
        r2 = 1 - mse / baseline_mse
        stored = summary['model']
        expected = [stored['mse']['mean'], stored['context_mean_mse']['mean'], stored['paper_r2']['mean']]
        if stored['mse']['count'] != n or not np.allclose([mse, baseline_mse, r2], expected,
                                                         rtol=2e-5, atol=1e-7):
            raise ValueError(f'Пересчёт разошёлся с сохранёнными метриками: {directory}')
        rows.append({'step': step, 'n': n, 'mse': mse, 'paper_r2': r2,
                     'is_final': step == config['train']['steps']})
    if not any(row['is_final'] for row in rows):
        raise ValueError(f'Ещё нет финальной оценки {split} для {run.name}')
    return rows


def score_history(series_dir, variants, seeds):
    """Recompute ID/OOD metrics from the active checkpoint sweep of each completed training."""
    import pandas as pd
    import yaml
    from garg_notebook_io import read_json, read_jsonl
    from icl_lab.config import digest

    reference = None
    frames = []
    for variant in variants:
        for seed in seeds:
            run = project_path(series_dir / variant / f'seed-{seed}', must_exist=True)
            config = yaml.safe_load(project_path(run / 'config.yaml', must_exist=True).read_text())
            state = read_json(run / 'status.json')
            manifest = read_json(run / 'manifest.json')
            bank_info = read_json(run / 'banks/manifest.json')
            if (config['experiment'] != 'kobayashi' or config['model']['variant'] != variant
                    or config['model_seed'] != seed or config['technical_smoke']
                    or state['state'] != 'completed' or state['step'] != config['train']['steps']
                    or not (digest(config) == manifest['config_hash'] == bank_info['config_hash'])):
                raise ValueError(f'Нужен завершённый совместимый запуск: {run}')
            identity = (bank_info['task_state_sha256'], bank_info['banks']['id_test']['sha256'],
                        bank_info['banks']['test']['sha256'], manifest['code_hash'])
            if reference is not None and identity != reference:
                raise ValueError('Для сравнения нужны общие учитель, тестовые эпизоды и версия кода.')
            reference = identity
            starts = [e for e in read_jsonl(run / 'events.jsonl') if e['event'] == 'start']
            cutoffs = {e['attempt_id']: min((later['resume_from_step'] for later in starts[i + 1:]),
                                            default=state['step']) for i, e in enumerate(starts)}
            for split in ['id_test', 'test']:
                rows = _checkpoint_scores(run, config, split, bank_info, manifest['code_hash'], cutoffs)
                frames.append(pd.DataFrame(rows).assign(variant=variant, seed=seed, split=split))
    return pd.concat(frames, ignore_index=True)


def probe_results(series_dir, variants, seeds):
    """Load OOD diagnostics for the completed run's last checkpoint and banks."""
    from garg_notebook_io import sha256_file
    import pandas as pd

    rows, missing = [], []
    controls = ('trained_probe', 'shuffled_label_probe', 'untrained_model_probe')
    for variant in variants:
        for seed in seeds:
            run = project_path(series_dir / variant / f'seed-{seed}', must_exist=True)
            status = json.loads(project_path(run / 'status.json', must_exist=True).read_text())
            if status['state'] != 'completed' or status['technical_smoke']:
                raise ValueError(f'Для диагностики нужен завершённый полный запуск: {run}')
            manifest = json.loads(project_path(run / 'manifest.json', must_exist=True).read_text())
            bank_info = json.loads(project_path(run / 'banks/manifest.json', must_exist=True).read_text())
            banks = {key: value['sha256'] for key, value in bank_info['banks'].items()}
            checkpoint = project_path(run / 'checkpoints/last.pt', must_exist=True)
            checksum = sha256_file(checkpoint)
            device = manifest['environment']['device']
            key = f"{status['step']:08d}-{checksum[:12]}-{manifest['code_hash'][:12]}-{device}"
            path = project_path(run / 'analysis' / key / 'diagnostics.json')
            if not path.is_file():
                missing.append(run)
                continue
            saved = json.loads(path.read_text())
            expected = {'step': status['step'], 'attempt_id': status['attempt_id'], 'bank_hashes': banks,
                        'checkpoint_sha256': checksum, 'code_hash': manifest['code_hash'],
                        'technical_smoke': False, 'device': device}
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Диагностика не соответствует весам и банкам: {path}')
            for control in controls:
                probe = saved['results'][control]
                test = probe['splits']['test']
                rows.append((variant, seed, control, probe['selected_layer'], probe['selected_ridge'],
                             test['count'], test['code_r2_macro'], test['executor_context_mean_r2'],
                             test['invalid_executor_count']))
    return pd.DataFrame(rows, columns=['variant', 'seed', 'control', 'layer', 'ridge', 'episodes',
                                       'code_r2', 'executor_r2', 'invalid']), missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["id_test", "test"], required=True)
    parser.add_argument("--row", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(_inspect(args.series, args.seed, args.split, args.row), allow_nan=False))
