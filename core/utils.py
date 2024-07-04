import os
import random
import shutil
import numpy as np

from core.typing import ModelPath, get_basic_model_name
from core.names import DL_LIB
from tools.log import do_logging
from tools.utils import modify_config
from tools import yaml_op


def configure_gpu(configs, idx=-1):
  if configs[0].dllib == 'jax':
    configure_jax_gpu(idx)
  else:
    device = configure_torch_gpu(idx)
    for config in configs:
      config = modify_config(config, device=device)
  return configs


def configure_jax_gpu(idx=-1):
  """ Configures gpu for Tensorflow/JAX
    The standard way described in the document of JAX does not work for TF. Since 
    we utilize the later for data visualization in Tensorboard, we 
  Args:
    idx: index(es) of PhysicalDevice objects returned by `list_physical_devices`
  """
  # if idx is not None and idx >= 0:
  #   os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
  import jax
  import tensorflow as tf
  # tf.config.experimental.set_visible_devices([], 'GPU')
  gpus = tf.config.list_physical_devices('GPU')
  n_gpus = len(gpus)
  if idx is None or n_gpus == 0:
    jax.config.update('jax_platforms', 'cpu')
    do_logging('No gpu is used', backtrack=3, color='red')
    return False
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  # restrict TensorFlow to only use the i-th GPU
  if idx >= 0:
    gpus = [gpus[idx % n_gpus]]
  if gpus:
    try:
      # memory growth
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.experimental.set_visible_devices(gpus, 'GPU')
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      do_logging(f'{n_gpus} Physical GPUs, {len(logical_gpus)} Logical GPU', color='red')
    except RuntimeError as e:
      # visible devices must be set before GPUs have been initialized
      do_logging(e, color='red')
    return True

def get_jax_device():
  import jax
  platform = jax.config.read('jax_platform_name')
  do_logging(f'platform name: {platform}', color='red')
  device = jax.devices()
  return device


def configure_torch_gpu(idx=-1):
  """ Configures gpu for Tensorflow/JAX
    The standard way described in the document of JAX does not work for TF. Since 
    we utilize the later for data visualization in Tensorboard, we 
  Args:
    idx: index(es) of PhysicalDevice objects returned by `list_physical_devices`
  """
  # if idx is not None and idx >= 0:
  #   os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
  import tensorflow as tf
  import torch
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  gpus = tf.config.list_physical_devices('GPU')
  n_gpus = len(gpus)
  if idx >= 0:
    gpus = [gpus[idx % n_gpus]]
  if gpus:
    try:
      # memory growth
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.experimental.set_visible_devices(gpus, 'GPU')
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      do_logging(f'{n_gpus} Physical GPUs, {len(logical_gpus)} Logical GPU', color='red')
    except RuntimeError as e:
      # visible devices must be set before GPUs have been initialized
      do_logging(e, color='red')
  n_gpus = torch.cuda.device_count()
  if idx >= 0:
    idx = idx % n_gpus
  device = f'cuda:{idx}' if idx >= 0 else 'cpu'
  return device

def get_num_gpus():
  import tensorflow as tf
  gpus = tf.config.list_physical_devices('GPU')
  n_gpus = len(gpus)
  return n_gpus

def set_seed(seed: int=None, dllib=DL_LIB.JAX):
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if dllib == DL_LIB.TORCH:
      import torch
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
  do_logging(f'seed={seed}', backtrack=3, level='info')

def save_code(model_path: ModelPath, backtrack=3):
  """ Saves the code so that we can check the chagnes latter """
  dest_dir = os.path.join(*model_path, 'src')
  if os.path.isdir(dest_dir):
    shutil.rmtree(dest_dir)
  
  shutil.copytree('.', dest_dir, 
    ignore=shutil.ignore_patterns(
      '*logs*', '*data*', '.*', '*.md',
      '*pycache*', '*.pyc', '*test*', '*outs*', 
      '*results*', '*.tar', '*__*'))
  do_logging(f'Save code: {model_path}', backtrack=backtrack, level='info')

def save_code_for_seed(config, seed=0):
  if config.seed == seed:
    root_dir = config.root_dir
    model_name = get_basic_model_name(config.model_name)
    save_code(ModelPath(root_dir, model_name), backtrack=4)

def save_config(config, model_path=None, config_name='config.yaml'):
  if model_path is None:
    model_path = ModelPath(config.root_dir, config.model_name)
  else:
    assert model_path.root_dir == config.root_dir, (model_path.root_dir, config.root_dir)
    assert model_path.model_name == config.model_name, (model_path.model_name, config.model_name)
  yaml_op.save_config(config, path=os.path.join(*model_path, config_name))
