import yaml
import numpy as np
from pathlib import Path

from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from tools.utils import eval_config, flatten_dict


YAML_SUFFIX = '.yaml'

def default_path(path):
  if path.startswith('/'):
    return Path(path)
  else:
    return Path('.') / path

def simplify_datatype(config):
  """ Converts ndarray to list, useful for saving config as a yaml file """
  if isinstance(config, AttrDict):
    config = config.asdict()
  if isinstance(config, (int, float, np.floating, np.integer, str)):
    return config
  if config is None:
    return config
  if isinstance(config, np.ndarray):
    return config.tolist()
  if isinstance(config, (list, tuple)):
    return [simplify_datatype(v) for v in config]
  try:
    for k, v in config.items():
      if isinstance(v, dict):
        config[k] = simplify_datatype(v)
      elif isinstance(v, (list, tuple)):
        config[k] = simplify_datatype(v)
      elif isinstance(v, np.ndarray):
        config[k] = v.tolist()
      else:
        config[k] = v
  except Exception as e:
    do_logging(str(config), color='red')
    do_logging(f'Exception: {e}', color='red')
    exit()
  return config

# load arguments from config.yaml
def load_config(path='config', to_attrdict=True, to_eval=True):
  if not path.endswith(YAML_SUFFIX):
    path = path + YAML_SUFFIX
  path = default_path(path)
  if not path.exists():
    do_logging(f'No configuration is found at: {path}', level='pwc', backtrack=4)
    return AttrDict()
  with open(path, 'r') as f:
    try:
      config = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
      do_logging(f'Fail loading configuration: {path}', level='pwc', backtrack=4)
      do_logging(f'Error message: {exc}', level='pwc', backtrack=4)
  if to_eval:
    config = eval_config(config)
  if to_attrdict:
    return dict2AttrDict(config)
  else:
    return config

# save config to config.yaml
def save_config(config: dict, config_to_update={}, path='config'):
  assert isinstance(config, dict)
  config = simplify_datatype(config)
  if not path.endswith(YAML_SUFFIX):
    path = path + YAML_SUFFIX
  
  path = default_path(path)
  if path.exists():
    if config_to_update is None:
      config_to_update = load_config(path)
  else:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

  with path.open('w') as f:
    try:
      config_to_update.update(config)
      yaml.dump(config_to_update, f)
    except yaml.YAMLError as exc:
      do_logging(f'Error message: {exc}', level='pwc', backtrack=4)

def load(path: str):
  if not Path(path).exists():
    return {}
  with open(path, 'r') as f:
    try:
      data = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
      do_logging(f'Fail loading configuration: {path}', level='pwc', backtrack=4)
      do_logging(f'Error message: {exc}', level='pwc', backtrack=4)
      return {}

  return data

def dump(path: str, config={}, **kwargs):
  if config:
    config = simplify_datatype(config)
    config = {'config': config, **kwargs}
  else:
    config = kwargs
  path = default_path(path)
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
  with path.open('w') as f:
    try:
      yaml.dump(config, f)
    except yaml.YAMLError as exc:
      do_logging(f'Error message: {exc}', level='pwc', backtrack=4)

def yaml2json(yaml_path, json_path, flatten=False):
  config = load_config(yaml_path)
  if flatten:
    config = flatten_dict(config)
  import json
  with open(json_path, 'w') as json_file:
    json.dump(config, json_file)

  return config
