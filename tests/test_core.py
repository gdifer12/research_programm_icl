from pathlib import Path
import copy
import json
import numpy as np
import pytest
import torch
from icl_lab import config
from icl_lab.engine import train, prepare, read_bank, evaluate, load_run, evaluate_model
from icl_lab.storage import load_torch, sha
from icl_lab.analysis import analyze, infer, report, accepted_rows
from icl_lab.tasks import make_task

ROOT=Path(__file__).parents[1]


def small(name):
    c=config.load(ROOT/f'configs/experiments/{name}.yaml',smoke=True)
    c['train'].update(steps=4,save_every=2,eval_every=2,log_every=1)
    c['eval'].update(episodes=12,diagnostic_episodes=12,batch_size=6,bootstrap_samples=10)
    return c


@pytest.mark.parametrize('name',['garg','assign_add','kobayashi','kobayashi_hyper'])
def test_exact_resume_and_complete_artifacts(name,tmp_path):
    c=small(name)
    whole,part=tmp_path/'whole',tmp_path/'part'
    train(c,whole)
    train(c,part,stop_after=2)
    assert json.loads((part/'status.json').read_text())['state']=='paused'
    train(c,part,resume='last')
    a,b=[load_torch(path/'checkpoints/last.pt',full=True) for path in [whole,part]]
    assert a['step']==b['step']==4
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key]),key
    assert torch.equal(a['rng']['train_generator'],b['rng']['train_generator'])
    assert load_torch(part/'checkpoints/previous.pt',full=True)['step']==2
    assert (part/'provenance/source.tar.gz').is_file()
    for split in ['id_test','test']:
        result=evaluate(part,'last',split)
        assert result['step']==4
        assert result==evaluate(part,'last',split)
    diagnostic=analyze(part,save_activations=True)
    assert diagnostic.is_file()
    inference=infer(part,'last',part/'inference.npz')
    assert inference.is_file()
    assert report(part).is_file()
    assert len(accepted_rows(part,'metrics.jsonl'))==4


def test_garg_leakage_baselines_and_curriculum():
    c=small('garg');task=make_task(c);model=task.build_model().eval()
    batch=task.sample(8,torch.Generator().manual_seed(1))
    before=task.predict(model,batch)
    modified={k:v.clone() for k,v in batch.items()}
    modified['y'][:,3:]+=100
    after=task.predict(model,modified)
    torch.testing.assert_close(before[:,:4],after[:,:4],rtol=0,atol=0)
    baselines=task.baseline_predictions(batch)
    torch.testing.assert_close(baselines['least_squares'][:,4:],batch['y'][:,4:],atol=1e-4,rtol=1e-4)
    assert torch.all(baselines['least_squares'][:,0]==0)
    c=config.load(ROOT/'configs/experiments/garg.yaml')
    task=make_task(c)
    first=task.sample(2,torch.Generator().manual_seed(1),step=0)
    last=task.sample(2,torch.Generator().manual_seed(1),step=30000)
    assert first['x'].shape==(2,11,20) and last['x'].shape==(2,41,20)
    assert not first['x'][:,:,5:].any()


@pytest.mark.parametrize('name',['garg','assign_add','kobayashi'])
def test_batch_invariance_and_bank_integrity(name,tmp_path):
    c=small(name);task=make_task(c);model=task.build_model().eval()
    prepare(c,tmp_path/'banks',task)
    bank,_=read_bank(tmp_path,'test')
    a=evaluate_model(task,model,bank,3,torch.device('cpu'))[0]
    b=evaluate_model(task,model,bank,12,torch.device('cpu'))[0]
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    path=tmp_path/'banks/test.npz';path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='hash'):read_bank(tmp_path,'test')


def test_protect_existing_runs_and_resume_config(tmp_path):
    c=small('garg');train(c,tmp_path/'run',stop_after=2)
    with pytest.raises(FileExistsError):train(c,tmp_path/'run')
    changed=copy.deepcopy(c);changed['train']['steps']+=1
    with pytest.raises(ValueError,match='same resolved config'):train(changed,tmp_path/'run',resume='last')
    with pytest.raises(ValueError,match='full resume'):train(c,tmp_path/'run',resume='initial')


@pytest.mark.parametrize('name',['garg','assign_add'])
def test_rollback_filters_evaluations_and_preserves_future_weights(name,tmp_path):
    from icl_lab.analysis import accepted_checkpoint
    c=small(name);run=tmp_path/'run'
    train(c,run);old=evaluate(run)
    assert accepted_checkpoint(run,old)
    report(run)
    plot_name='context-curve.png' if name=='garg' else 'groups.png'
    assert (run/'reports'/plot_name).is_file()
    train(c,run,resume='previous',stop_after=1)
    assert not accepted_checkpoint(run,old)
    assert list((run/'checkpoints/abandoned').rglob('final.pt'))
    report(run)
    assert not (run/'reports'/plot_name).exists()
    assert plot_name not in (run/'reports/summary.md').read_text()
    assert not json.loads((run/'reports/summary.json').read_text())['evaluations']
    new=evaluate(run)
    assert accepted_checkpoint(run,new)
    report(run)
    summary=json.loads((run/'reports/summary.json').read_text())
    assert [e['step'] for e in summary['evaluations']]==[3]


def test_corrupt_last_recovers_from_previous(tmp_path):
    c=small('garg');run=tmp_path/'run';train(c,run)
    last=run/'checkpoints/last.pt';last.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):train(c,run,resume='last')
    train(c,run,resume='previous')
    assert load_torch(last,full=True)['step']==4


@pytest.mark.parametrize('failed_replace',[1,2,3,4])
def test_checkpoint_rotation_always_leaves_a_valid_resume(tmp_path,monkeypatch,failed_replace):
    from icl_lab import engine
    c=small('garg');run=tmp_path/'run';train(c,run,stop_after=2)
    real_replace=engine.os.replace
    counter={'n':0}
    def interrupt(src,dst):
        dest=Path(dst)
        if dest.parent==run/'checkpoints' and dest.name in {'previous.pt','previous.pt.sha256','last.pt','last.pt.sha256'}:
            counter['n']+=1
            if counter['n']==failed_replace:raise OSError('Injected checkpoint interruption')
        return real_replace(src,dst)
    monkeypatch.setattr(engine.os,'replace',interrupt)
    with pytest.raises(OSError,match='Injected'):train(c,run,resume='last')
    valid=[]
    for name in ['last','previous']:
        try:valid.append(load_torch(run/f'checkpoints/{name}.pt',full=True)['step'])
        except (ValueError,FileNotFoundError):pass
    assert valid


def test_optional_corrupt_snapshot_does_not_block_resume(tmp_path):
    c=small('garg');run=tmp_path/'run';train(c,run,stop_after=2)
    optional=next((run/'checkpoints').glob('step-*.pt'))
    optional.with_suffix('.pt.sha256').unlink()
    train(c,run,resume='last')
    assert load_torch(run/'checkpoints/last.pt',full=True)['step']==4
    assert list((run/'checkpoints/abandoned').rglob(optional.name))


def test_rollback_previous_is_selected_ancestor(tmp_path):
    c=small('garg');run=tmp_path/'run';train(c,run)
    train(c,run,resume='previous',stop_after=1)
    assert load_torch(run/'checkpoints/last.pt',full=True)['step']==3
    assert load_torch(run/'checkpoints/previous.pt',full=True)['step']==2


def test_source_snapshot_without_git(tmp_path,monkeypatch):
    from icl_lab import storage
    original=storage.subprocess.run
    def without_git(command,*args,**kwargs):
        if command[0]=='git':raise FileNotFoundError('git is not installed')
        return original(command,*args,**kwargs)
    monkeypatch.setattr(storage.subprocess,'run',without_git)
    storage.snapshot(tmp_path/'source')
    meta=json.loads((tmp_path/'source/source.json').read_text())
    assert meta['git']=={'commit':None,'status':None,'diff':None}
    assert meta['archive_sha256']==sha(tmp_path/'source/source.tar.gz')
    assert 'src/icl_lab/engine.py' in meta['files']


def test_suite_stops_before_evaluation_when_training_pauses(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from icl_lab import __main__ as cli
    spec=tmp_path/'suite.yaml'
    spec.write_text('experiments:\n  - config: configs/experiments/garg.yaml\n    seeds: [1, 2]\n')
    calls=[]
    def paused_train(*args,**kwargs):
        calls.append(args[1])
        return {'state':'paused','step':1}
    def forbidden_evaluation(*args,**kwargs):
        pytest.fail('A paused suite must stop before evaluating or starting the next run')
    monkeypatch.setattr(cli.engine,'train',paused_train)
    monkeypatch.setattr(cli.engine,'evaluate',forbidden_evaluation)
    preview=cli.suite(SimpleNamespace(config=spec,output=tmp_path/'runs',device='cpu',
                                     smoke=True,execute=False,resume=False))
    assert all(command[0]==str(ROOT/'scripts/local-python') for command in preview['commands'])
    result=cli.suite(SimpleNamespace(config=spec,output=tmp_path/'runs',device='cpu',
                                    smoke=True,execute=True,resume=False))
    assert result['state']=='paused' and result['completed_runs']==0 and len(calls)==1


def test_periodic_analysis_checkpoints_are_retained(tmp_path):
    c=small('garg');c['train'].update(analysis_checkpoints=1,analysis_every=2)
    run=tmp_path/'run';train(c,run)
    assert [path.name for path in sorted((run/'checkpoints').glob('step-*.pt'))]==[
        'step-00000001.pt','step-00000002.pt','step-00000004.pt']
