from importlib import import_module


def make_task(config):
    name = config['experiment']
    if name not in {'garg', 'assign_add', 'kobayashi'}:
        raise ValueError(f'Unknown experiment: {name}')
    return import_module(f'icl_lab.tasks.{name}').Task(config)
