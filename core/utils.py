import os, shutil
import random
import numpy as np

from core.log import do_logging
from core.typing import ModelPath, AttrDict
from tools import yaml_op


def configure_gpu(idx=-1):
    """ Configures gpu for Tensorflow/JAX
        The standard way described in the document of JAX does not work for TF. Since 
        we utilize the later for data visualization in Tensorboard, we 
    Args:
        idx: index(es) of PhysicalDevice objects returned by `list_physical_devices`
    """
    # if idx is not None and idx >= 0:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import tensorflow as tf
    if idx is None:
        tf.config.experimental.set_visible_devices([], 'GPU')
        do_logging('No gpu is used', backtrack=3)
        return False
    gpus = tf.config.list_physical_devices('GPU')
    n_gpus = len(gpus)
    # restrict TensorFlow to only use the i-th GPU
    if idx >= 0:
        gpus = [gpus[idx % n_gpus]]
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    if gpus:
        try:
            # memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            do_logging(f'{n_gpus} Physical GPUs, {len(logical_gpus)} Logical GPU')
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            do_logging(e)
        return True

def set_seed(seed: int=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    do_logging(f'seed={seed}', backtrack=3)

def save_code(model_path: ModelPath):
    """ Saves the code so that we can check the chagnes latter """
    dest_dir = '/'.join([*model_path, 'src'])
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    
    shutil.copytree('.', dest_dir, 
        ignore=shutil.ignore_patterns(
            '*logs*', '*data*', '.*', '*.md',
            '*pycache*', '*.pyc', '*test*', '*outs*', 
            '*results*', '*env*', '*.tar', '*__*'))
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
        path='/'.join([*model_path, config_name]))

def get_vars_for_modules(modules):
    return sum([m.variables for m in modules], ())
