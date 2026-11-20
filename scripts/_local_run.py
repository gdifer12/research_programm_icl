"""Run the native project-local pipeline; MPS is the default device."""
import argparse
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from icl_lab.runtime import ROOT, project_path, require_local_environment


def main():
    require_local_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--allow-code-change', action='store_true')
    parser.add_argument('--allow-device-change', action='store_true')
    parser.add_argument('--analyze-all', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    def local(value, label, exists=False):
        path = Path(value)
        return project_path(path if path.is_absolute() else ROOT/path, label, exists)

    run = local(args.run_dir, 'run directory')
    config = local(args.config, 'configuration', True) if args.config else None
    if not config and not args.resume:
        parser.error('A new run requires --config')
    if args.resume and not run.is_dir():
        parser.error('--resume requires an existing run directory')
    if run.exists() and not run.is_dir():
        parser.error('--run-dir must be a directory')
    if run.is_dir() and any(run.iterdir()) and not args.resume:
        parser.error('Run directory is not empty; use --resume or a new directory')
    if args.resume and any(value is not None for value in [args.steps, args.seed]):
        parser.error('--steps/--seed apply to new runs; resume uses the saved configuration')
    if args.resume and args.smoke and config is None:
        parser.error('Resume already uses saved smoke settings; omit --smoke')
    if any(value is not None and value <= 0 for value in [args.steps, args.stop_after]):
        parser.error('--steps and --stop-after must be positive')
    resume = args.resume
    if resume and resume not in {'last', 'previous', 'initial', 'final'}:
        candidate = Path(resume)
        if not candidate.is_absolute():
            candidate = run/'checkpoints'/candidate
        resume = str(project_path(candidate, 'resume checkpoint', True))

    prefix = [str(ROOT/'scripts/local-python'), '-m', 'icl_lab']
    train = prefix + ['train', '--run-dir', str(run), '--device', args.device]
    if config:
        train += ['--config', str(config)]
    for option, value in [('resume', resume), ('steps', args.steps), ('stop-after', args.stop_after), ('seed', args.seed)]:
        if value is not None:
            train += ['--'+option, str(value)]
    if args.smoke:
        train.append('--smoke')
    if args.allow_code_change:
        train.append('--allow-code-change')
    if args.allow_device_change:
        train.append('--allow-device-change')
    commands = [prefix + ['doctor', '--device', args.device], train]
    commands += [prefix + ['evaluate', '--run-dir', str(run), '--checkpoint', 'all', '--split', split,
                           '--device', args.device] for split in ['id_test', 'test']]
    commands += [prefix + ['analyze', '--run-dir', str(run), '--checkpoint', 'all' if args.analyze_all else 'last',
                           '--device', args.device], prefix + ['report', '--run-dir', str(run)]]
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        print('# Evaluation and analysis run only after completed training.')
        return

    active = {'process': None, 'signal': None}

    def forward(signum, _frame):
        active['signal'] = signum
        child = active['process']
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, forward)
    for index, command in enumerate(commands):
        if active['signal'] is not None:
            raise SystemExit(128 + active['signal'])
        print('+ '+shlex.join(command), flush=True)
        child = subprocess.Popen(command, cwd=ROOT)
        active['process'] = child
        if active['signal'] is not None:
            forward(active['signal'], None)
        try:
            code = child.wait()
        finally:
            active['process'] = None
        if code:
            raise SystemExit(code if code > 0 else 128-code)
        if index == 1:
            state = json.loads(project_path(run/'status.json', 'training status', True).read_text())['state']
            if state == 'paused':
                continuation = [str(ROOT/'scripts/local-run'), '--run-dir', str(run), '--resume', 'last',
                                '--device', args.device]
                if args.analyze_all:
                    continuation.append('--analyze-all')
                print('Paused with a full checkpoint. Continue:\n'+shlex.join(continuation))
                return
            if state != 'completed':
                raise RuntimeError(f'Training state {state!r}; later stages were not started')


if __name__ == '__main__':
    main()
