import os, shutil
import numpy as np

from core.log import do_logging
from core.typing import ModelPath, AttrDict
from tools import yaml_op


def save_code(model_path: ModelPath):
    """ Saves the code so that we can check the chagnes latter """
    dest_dir = '/'.join([*model_path, 'src'])
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    
    shutil.copytree('.', dest_dir, 
        ignore=shutil.ignore_patterns(
            '*logs*', '*data*', '.*', '*.md',
            '*pycache*', '*.pyc', '*test*', '*outs*', 
            '*results*', '*env*'))
    do_logging(
        f'Save code: {model_path}', 
        level='print', 
        time=True, 
        backtrack=3, 
    )

def simplify_datatype(config):
    """ Converts ndarray to list, useful for saving config as a yaml file """
    if isinstance(config, AttrDict):
        config = config.asdict()
    for k, v in config.items():
        if isinstance(v, dict):
            config[k] = simplify_datatype(v)
        elif isinstance(v, tuple):
            config[k] = list(v)
        elif isinstance(v, np.ndarray):
            config[k] = v.tolist()
        else:
            config[k] = v
    return config

def save_config(config, model_path=None, config_name='config.yaml'):
    if model_path is None:
        model_path = ModelPath(config.root_dir, config.model_name)
    else:
        assert model_path.root_dir == config.root_dir, (model_path.root_dir, config.root_dir)
        assert model_path.model_name == config.model_name, (model_path.model_name, config.model_name)
    config = simplify_datatype(config)
    yaml_op.save_config(config, 
        filename='/'.join([*model_path, config_name]))

def get_vars_for_modules(modules):
    return sum([m.variables for m in modules], ())
