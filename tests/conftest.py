"""Keep every pytest temporary artifact inside the project as well."""
import os
from icl_lab.runtime import ROOT, configure_process, project_path


def pytest_configure(config):
    configure_process()
    requested = config.option.basetemp
    config.option.basetemp = str(project_path(requested or ROOT / 'artifacts/tmp' / f'pytest-{os.getpid()}',
                                              'pytest temporary directory'))
