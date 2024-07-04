import os, shutil
import random
import numpy as np
import torch

from tools.log import do_logging
from core.typing import ModelPath, get_basic_model_name
from tools import yaml_op


def tpdv(device):
  return dict(dtype=torch.float32, device=torch.device(device))

def set_seed(seed: int=None):
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
      '*results*', '*env*', '*.tar', '*__*'))
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
  yaml_op.save_config(config, path = os.path.join(*model_path, config_name))
