"""Project-local paths and process configuration, before importing ML libraries.

This constrains application paths; it is not an operating-system sandbox.
macOS, Metal and system tools still use their installed system components.
"""
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def project_path(value, label='path', must_exist=False):
    """Reject traversal and symlinks escaping the portable icl project root."""
    path = Path(value).absolute().resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'{label} must stay inside {ROOT}: {value}')
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def configure_process():
    paths = {
        'TMPDIR': ROOT / 'artifacts/tmp',
        'TMP': ROOT / 'artifacts/tmp',
        'TEMP': ROOT / 'artifacts/tmp',
        'XDG_CACHE_HOME': ROOT / 'artifacts/cache',
        'XDG_CONFIG_HOME': ROOT / 'artifacts/config',
        'XDG_DATA_HOME': ROOT / 'artifacts/local-data',
        'MPLCONFIGDIR': ROOT / 'artifacts/cache/matplotlib',
        'TORCH_HOME': ROOT / 'artifacts/cache/torch',
        'HF_HOME': ROOT / 'artifacts/cache/huggingface',
        'PIP_CACHE_DIR': ROOT / 'artifacts/cache/pip',
        'PYTHONPYCACHEPREFIX': ROOT / 'artifacts/cache/pycache',
    }
    for key, value in paths.items():
        directory = project_path(value, label=key)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    tempfile.tempdir = os.environ['TMPDIR']
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['PIP_CONFIG_FILE'] = os.devnull
    os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    return paths


def require_local_environment():
    """A venv is insufficient if its base interpreter lives outside the project."""
    for label, value in [('Python executable', sys.executable), ('environment', sys.prefix),
                         ('base interpreter', sys.base_prefix)]:
        project_path(value, label=label, must_exist=True)
    if sys.flags.no_user_site != 1:
        raise RuntimeError('Use scripts/local-python (Python -s, no user site packages)')
