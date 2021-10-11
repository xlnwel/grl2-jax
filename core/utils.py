import os, shutil
import numpy as np

from utility import yaml_op
from utility.typing import AttrDict


def save_code(root_dir, model_name):
    """ Saves the code so that we can check the chagnes latter """
    dest_dir = f'{root_dir}/{model_name}/src'
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    
    shutil.copytree('.', dest_dir, 
        ignore=shutil.ignore_patterns(
            '*logs*', 'data/*', '.*', '*.md',
            '*pycache*', '*.pyc', '*test*',
            '*results*'))

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

def save_config(root_dir, model_name, config):
    config = simplify_datatype(config)
    yaml_op.save_config(config, filename=f'{root_dir}/{model_name}/config.yaml')
