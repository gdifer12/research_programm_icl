import argparse
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
import yaml
from . import config
from . import engine
from . import analysis
from .devices import require_device, synchronize
from .runtime import project_path, require_local_environment
from .storage import ROOT, clean, environment, lock, write_json, load_torch


class Tee:
    def __init__(self, console, file):self.console,self.file=console,file
    def write(self, text):self.console.write(text);self.file.write(text);self.file.flush();return len(text)
    def flush(self):self.console.flush();self.file.flush()


def doctor(device, output=None):
    dev=require_device(device)
    if dev.type=='cuda':
        cap=torch.cuda.get_device_capability(dev)
        if cap==(7,0) and 'sm_70' not in torch.cuda.get_arch_list():
            raise RuntimeError('This PyTorch wheel lacks V100 sm_70 support')
    x=torch.ones((8,8),device=dev,requires_grad=True)
    loss=(x@x).square().mean();loss.backward()
    synchronize(dev)
    if not torch.isfinite(x.grad).all():raise RuntimeError('Nonfinite device smoke')
    result=environment(dev);result['forward_backward_passed']=True
    if output:write_json(output,result)
    return result


def suite(args):
    specification=yaml.safe_load(project_path(args.config, 'suite configuration', must_exist=True).read_text())
    root=project_path(args.output, 'suite output')
    plans=[]
    for entry in specification['experiments']:
        path=Path(entry['config'])
        if not path.is_absolute():path=ROOT/path
        path=project_path(path, 'experiment configuration', must_exist=True)
        for seed in entry.get('seeds',[1]):
            run=project_path(root/f'{path.stem}-seed{seed}', 'run directory')
            command=[str(ROOT/'scripts/local-python'),'-m','icl_lab','train','--config',str(path),'--run-dir',str(run),'--device',args.device,'--seed',str(seed)]
            if args.smoke:command.append('--smoke')
            plans.append((path,seed,run,command))
    if not args.execute:
        return {'execution':'dry-run','sequential':True,'commands':[p[3] for p in plans]}
    root.mkdir(parents=True,exist_ok=True)
    with lock(root):
        records=[]
        for path,seed,run,command in plans:
            c=config.load(path,smoke=args.smoke,seed=seed)
            resume='last' if args.resume and project_path(run/'config.yaml', 'run configuration').exists() else None
            status=engine.train(c,run,args.device,resume=resume,
                                allow_device_change=getattr(args,'allow_device_change',False))
            if status['state']!='completed':
                return {'state':status['state'],'paused_run':str(run),'completed_runs':len(records),
                        'output':str(root),'sequential':True,'resume_with':'suite --execute --resume with the same arguments'}
            evaluations={split:engine.evaluate(run,'last',split,args.device) for split in ['id_test','test']}
            analysis.analyze(run,'last',args.device)
            analysis.report(run)
            records.append({'config':str(path.relative_to(ROOT)),'seed':seed,'run':str(run),'evaluations':evaluations})
            write_json(root/'suite.json',{'technical_smoke':args.smoke,'runs':records})
        aggregate={}
        for row in records:
            group=aggregate.setdefault(row['config'],{})
            for split,result in row['evaluations'].items():
                for metric,values in result['model'].items():
                    if values['mean'] is not None:
                        group.setdefault(split+'/'+metric,[]).append(values['mean'])
        summary={name:{metric:{'runs':len(values),'mean':np.nanmean(values,axis=0).tolist(),
                   'std_across_runs':np.nanstd(values,axis=0,ddof=1).tolist() if len(values)>1 else None}
                 for metric,values in group.items()} for name,group in aggregate.items()}
        write_json(root/'aggregate.json',summary)
        return {'runs':len(records),'output':str(root),'sequential':True}


def checkpoint_sweep(run, command, split, device, save_activations=False):
    root = project_path(run, 'run directory')
    directory = project_path(root/'checkpoints', 'checkpoint directory')
    paths = sorted(directory.glob('step-*.pt'))
    paths.insert(0, directory/'initial.pt')
    results = []
    for path in paths:
        saved = load_torch(path, full=True)
        if not analysis.accepted_checkpoint(root, saved):
            continue
        if command == 'evaluate':
            result = engine.evaluate(root, str(path), split, device)
            results.append({'step':result['step'], 'split':split})
        else:
            results.append(str(analysis.analyze(root, str(path), device, save_activations)))
    return {'checkpoints_processed':len(results), 'results':results}


def main():
    parser=argparse.ArgumentParser(description='ICL experiments: explicit local files, no automatic cluster submission')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('doctor');p.add_argument('--device',default='mps');p.add_argument('--output')
    p=sub.add_parser('prepare');p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('train');p.add_argument('--config');p.add_argument('--run-dir',required=True);p.add_argument('--device',default='mps')
    p.add_argument('--resume');p.add_argument('--smoke',action='store_true');p.add_argument('--steps',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--stop-after',type=int);p.add_argument('--allow-code-change',action='store_true')
    p.add_argument('--allow-device-change',action='store_true',help='Explicitly continue on another backend; this is not an identical continuation')
    for name in ['evaluate','analyze','infer']:
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True);p.add_argument('--checkpoint',default='last');p.add_argument('--device',default='mps')
        if name in {'evaluate','infer'}:p.add_argument('--split',choices=engine.SPLITS,default='test')
        if name=='analyze':p.add_argument('--save-activations',action='store_true')
        if name=='infer':p.add_argument('--input');p.add_argument('--output',required=True)
    p=sub.add_parser('report');p.add_argument('--run-dir',required=True)
    p=sub.add_parser('suite');p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='mps')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--allow-device-change',action='store_true')
    args=parser.parse_args()
    require_local_environment()
    for name in ['config','input','output','run_dir']:
        value=getattr(args,name,None)
        if value is not None:
            setattr(args,name,project_path(value,name,must_exist=name in {'config','input'}))
    for name in ['checkpoint','resume']:
        value=getattr(args,name,None)
        if value is not None and value not in {'last','previous','initial','final','all'}:
            # A simple filename is resolved by checkpoint_path below run/checkpoints.
            # Any provided path must already be within this project.
            project_path(value,name)
    if args.command=='doctor':result=doctor(args.device,args.output)
    elif args.command=='prepare':result=engine.prepare(config.load(args.config,args.smoke),args.output)
    elif args.command=='train':
        if not args.config and not args.resume:parser.error('new train requires --config')
        if args.stop_after is not None and args.stop_after<=0:parser.error('--stop-after must be positive')
        c=config.load(args.config,args.smoke,args.steps,args.seed) if args.config else None
        # stdout.log lives beside the run; opening it must not pre-create a nonempty new run.
        run=project_path(args.run_dir,'run directory');run.parent.mkdir(parents=True,exist_ok=True)
        log=project_path(run.parent/(run.name+'.stdout.log'),'console log')
        with open(log,'a') as stream, redirect_stdout(Tee(sys.stdout,stream)), redirect_stderr(Tee(sys.stderr,stream)):
            result=engine.train(c,run,args.device,args.resume,args.stop_after,args.allow_code_change,args.allow_device_change)
    elif args.command=='evaluate':
        result=checkpoint_sweep(args.run_dir,'evaluate',args.split,args.device) if args.checkpoint=='all' else engine.evaluate(args.run_dir,args.checkpoint,args.split,args.device)
    elif args.command=='analyze':
        result=checkpoint_sweep(args.run_dir,'analyze',None,args.device,args.save_activations) if args.checkpoint=='all' else str(analysis.analyze(args.run_dir,args.checkpoint,args.device,args.save_activations))
    elif args.command=='infer':result=str(analysis.infer(args.run_dir,args.checkpoint,args.output,args.device,args.input,args.split))
    elif args.command=='report':result=str(analysis.report(args.run_dir))
    elif args.command=='suite':result=suite(args)
    print(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
