from pathlib import Path
import csv
import json
import os
import numpy as np
import torch
import yaml
from .engine import load_run, move, read_bank, evaluate_model
from .storage import lock, save_npz, sha, write_json, code_hash
from .runtime import project_path, configure_process


def analyze(run, checkpoint='last', device='cpu', save_activations=False):
    run = project_path(run, 'run directory', must_exist=True)
    with lock(run):
        config, task, model, saved, path, device = load_run(run, checkpoint, device)
        n = config['eval']['diagnostic_episodes']
        banks, hashes = {}, {}
        for split in ['probe_train', 'validation', 'id_test', 'test']:
            bank, hashes[split] = read_bank(run, split)
            banks[split] = {k: v[:n] for k, v in bank.items()}
        key = sha(path)[:12] + '-' + code_hash()[:12] + '-' + device.type
        output = project_path(run / 'analysis' / f'{saved["step"]:08d}-{key}', 'analysis output')
        output.mkdir(parents=True, exist_ok=True)
        result = task.diagnostics(model, banks, device)
        if save_activations:
            # Explicit bounded export, never every layer of every training update.
            with torch.no_grad():
                values = task.activations(model, move(banks['test'], device))
            save_npz(output / 'activations.npz', values)
        write_json(output / 'diagnostics.json', {'step': saved['step'], 'attempt_id': saved['attempt_id'], 'checkpoint_sha256': sha(path),
                    'bank_hashes': hashes, 'code_hash': code_hash(), 'device': str(device), 'episodes_per_bank_limit': n,
                    'technical_smoke': config.get('technical_smoke', False), 'results': result})
        return output / 'diagnostics.json'


def infer(run, checkpoint, output, device='cpu', input_path=None, split='test'):
    from .storage import load_npz
    output = project_path(output, 'inference output')
    if input_path is not None:
        input_path = project_path(input_path, 'inference input', must_exist=True)
    config, task, model, saved, path, device = load_run(run, checkpoint, device)
    if input_path is not None:
        bank = load_npz(input_path)
        bank_hash = sha(input_path)
    else:
        bank, bank_hash = read_bank(run, split)
    predictions = []
    n = next(iter(bank.values())).shape[0]
    with torch.no_grad():
        for start in range(0, n, config['eval']['batch_size']):
            batch = move({k: v[start:start+config['eval']['batch_size']] for k, v in bank.items()}, device)
            predictions.append(task.predict(model, batch).detach().cpu())
    pred = torch.cat(predictions)
    if not torch.isfinite(pred).all():
        raise FloatingPointError('Nonfinite inference output')
    if output.exists():
        raise FileExistsError(output)
    save_npz(output, {'prediction': pred, 'episode_id': bank.get('episode_id', torch.arange(n))})
    write_json(output.with_suffix('.json'), {'checkpoint_sha256': sha(path), 'bank_sha256': bank_hash,
                                           'step': saved['step'], 'code_hash': code_hash(), 'device': str(device), 'episodes': n})
    return output


def accepted_rows(run, filename):
    """Preserve raw logs while discarding abandoned tails after an older resume."""
    path = project_path(Path(run) / filename, 'run log')
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = [json.loads(line) for line in project_path(Path(run)/'events.jsonl').read_text().splitlines() if line.strip()]
    accepted = []
    for event in events:
        if event.get('event') == 'start':
            accepted = [r for r in accepted if r['step'] <= event['resume_from_step']]
            accepted.extend(r for r in rows if r['attempt_id'] == event['attempt_id'])
    return sorted(accepted, key=lambda row: row['step'])


def accepted_checkpoint(run, saved):
    events = [json.loads(line) for line in project_path(Path(run)/'events.jsonl').read_text().splitlines() if line.strip()]
    starts = [event for event in events if event.get('event') == 'start']
    for i, event in enumerate(starts):
        if event['attempt_id'] == saved.get('attempt_id'):
            return all(saved['step'] <= later['resume_from_step'] for later in starts[i+1:])
    return False


def report(run):
    run = project_path(run, 'report run', must_exist=True)
    with lock(run):
        output = project_path(run / 'reports', 'report output')
        output.mkdir(exist_ok=True)
        configure_process()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        config = yaml.safe_load(project_path(run/'config.yaml').read_text())
        title_prefix = 'SMOKE — technical check | ' if config.get('technical_smoke') else ''
        status_path = project_path(run/'status.json')
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        summary_paths = [project_path(p, 'evaluation summary') for p in project_path(run/'evaluations').glob('*/summary.json')]
        summaries = [json.loads(p.read_text()) for p in sorted(summary_paths, key=lambda p:p.stat().st_mtime)]
        summaries = [s for s in summaries if accepted_checkpoint(run, s)]
        # Re-evaluation retains all artifacts; last result per split and step is the active view.
        by_key = {(s['split'], s['step']): s for s in summaries}
        summaries = list(by_key.values())
        rows = []
        for s in summaries:
            methods = {'model': s['model'], **s['baselines']}
            for method, metrics in methods.items():
                for metric, item in metrics.items():
                    values = item['mean'] if isinstance(item['mean'], list) else [item['mean']]
                    counts = item.get('count')
                    for i, value in enumerate(values):
                        rows.append({'step':s['step'], 'split':s['split'], 'method':method, 'metric':metric,
                                     'coordinate':i if len(values)>1 else '', 'mean':value,
                                     'count':counts[i] if isinstance(counts,list) else counts})
        with open(project_path(output/'metrics.csv'), 'w', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=['step','split','method','metric','coordinate','mean','count'])
            writer.writeheader();writer.writerows(rows)
        logs=accepted_rows(run, 'metrics.jsonl')
        validation=accepted_rows(run, 'validation.jsonl')
        # These are derived views. A rollback may leave no accepted data to redraw
        # a plot, so never reuse an image from the previously rendered branch.
        plot_names = ['training.png','validation.png','context-curve.png','groups.png']
        for name in plot_names:
            project_path(output/name).unlink(missing_ok=True)
        if logs:
            fig, ax=plt.subplots(figsize=(7,4))
            ax.plot([r['step'] for r in logs],[r['train_loss'] for r in logs])
            ax.set(xlabel='Completed optimizer updates',ylabel='Training objective',title=title_prefix+config['experiment'])
            ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'training.png',dpi=160);plt.close(fig)
        if validation:
            fig,ax=plt.subplots(figsize=(8,4))
            keys=sorted({k for row in validation for k in row['metrics'] if
                         (config['experiment']=='garg' and k=='nmse') or
                         (config['experiment']=='assign_add' and k.startswith('accuracy/')) or
                         (config['experiment']=='kobayashi' and k=='paper_r2')})
            for key in keys:
                series=[]
                for row in validation:
                    mean=row['metrics'].get(key,{}).get('mean')
                    series.append(np.nan if mean is None else float(np.nanmean(np.asarray(mean,dtype=float))))
                ax.plot([r['step'] for r in validation],series,label=key)
            ax.set(xlabel='Completed optimizer updates',ylabel='ID validation metric',title=title_prefix+config['experiment'])
            ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'validation.png',dpi=160);plt.close(fig)
        if config['experiment']=='garg' and summaries:
            s=max((s for s in summaries if s['split']=='test'),key=lambda s:s['step'],default=summaries[-1])
            fig,ax=plt.subplots(figsize=(7,4))
            for name, metrics in {'model':s['model'],**s['baselines']}.items():
                if 'nmse' in metrics:ax.plot(metrics['nmse']['mean'],label=name)
            if 'nmse_bootstrap_90' in s:
                lo,hi=s['nmse_bootstrap_90'];ax.fill_between(range(len(lo)),lo,hi,alpha=.15)
            ax.set(xlabel='Context examples k',ylabel='MSE / dimension',title=title_prefix+f'Garg, step {s["step"]}')
            ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'context-curve.png',dpi=160);plt.close(fig)
        elif summaries:
            last_step=max(s['step'] for s in summaries)
            bars=[r for r in rows if r['step']==last_step and r['method']=='model' and r['mean'] is not None
                  and (r['metric'].startswith('accuracy/') or r['metric']=='paper_r2')]
            if bars:
                fig,ax=plt.subplots(figsize=(10,5))
                ax.bar(range(len(bars)),[r['mean'] for r in bars])
                ax.set_xticks(range(len(bars)),[r['split']+':'+r['metric'] for r in bars],rotation=55,ha='right',fontsize=8)
                ax.set_title(title_prefix+f'{config["experiment"]}, step {last_step}');fig.tight_layout();fig.savefig(output/'groups.png',dpi=160);plt.close(fig)
        label='ТЕХНИЧЕСКИЙ SMOKE: не научный результат.' if config.get('technical_smoke') else 'Оценки требуют содержательного сопоставления с протоколом статьи.'
        text=f'# {config["experiment"]}\n\n{label}\n\nСостояние: {status.get("state","unknown")}; updates={status.get("step",0)}.\n\n'
        text+='[Все метрики](metrics.csv). Средние по эпизодам одного запуска не заменяют разброс между независимыми обучениями.\n\n'
        for name in plot_names:
            if (output/name).exists():text+=f'![{name}]({name})\n\n'
        project_path(output/'summary.md').write_text(text)
        write_json(output/'summary.json',{'experiment':config['experiment'],'status':status,'evaluations':summaries,
                   'accepted_training_rows':len(logs),'technical_smoke':config.get('technical_smoke',False)})
        return output/'summary.md'
