"""Run a notebook's CPU inspection in the kernel's Python against this checkout."""
import json
import os
import subprocess
import sys

from icl_lab.runtime import ROOT


def inspect_json(script, arguments):
    checkout_paths = os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'scripts')))
    environment = dict(os.environ, PYTHONPATH=checkout_paths, PYTHONNOUSERSITE='1')
    command = [sys.executable, '-s', str(script), *arguments]
    return json.loads(subprocess.check_output(command, cwd=ROOT, env=environment, text=True))
