import os
import logging

from core.log import do_logging
from core.names import PATH_SPLIT
from core.typing import dict2AttrDict
from tools import pkg
from tools.file import search_for_all_files, search_for_file
from tools.utils import eval_str, modify_config
from tools.yaml_op import load_config

logger = logging.getLogger(__name__)


def get_configs_dir(algo):
  algo_dir = pkg.get_package_from_algo(algo, 0, PATH_SPLIT)
  if algo_dir is None:
    raise RuntimeError(f'Algorithm({algo}) is not implemented')
  configs_dir = os.path.join(algo_dir, 'configs')

  return configs_dir


def get_filename_with_env(env):
  env_split = env.split('-', 1)
  if len(env_split) > 1:
    filename = env_split[0]
  elif len(env_split) == 1:
    filename = 'gym'
  else:
    raise ValueError(f'Cannot extract filename from env: {env}')

  return filename


def change_config_with_key_value(config, key, value, prefix=''):
  modified_configs = []
  original_key = key
  if ':' in key:
    keys = key.split(':')
    key = keys[0]
  else:
    keys = None
  if isinstance(value, str) and ',' in value:
    value = [eval_str(v) for v in value.split(',')]
  for k, v in config.items():
    config_name = f'{prefix}:{k}' if prefix else k
    if key == k:
      if keys is None:
        config[k] = value
        modified_configs.append(config_name)
      else:
        keys_in_config = True
        key_config = config[k]
        for kk in keys[1:-1]:
          if kk not in key_config:
            keys_in_config = False
            break
          key_config = key_config[kk]
          config_name = f'{config_name}:{kk}'
        if keys_in_config and keys[-1] in key_config:
          key_config[keys[-1]] = value
          config_name = f'{config_name}:{keys[-1]}'
          modified_configs.append(config_name)
    if isinstance(v, dict):
      modified_configs += change_config_with_key_value(
        v, original_key, value, config_name)

  return modified_configs


def change_config_with_kw_string(kw, config, config_idx=None):
  """ Changes configs based on kw. model_name will
  be modified accordingly to embody changes 
  """
  if kw:
    for s in kw:
      key, value = s.split('=', 1)
      config[key] = value
      if '#' in key:
        i, key = key.split('#')
        modify_config = False
        if ',' in i:
          for ii in i.split(','):
            if eval_str(ii) == config_idx:
              modify_config = True
        elif eval_str(i) == config_idx:
          modify_config = True
        if not modify_config:
          continue
      value = eval_str(value)

      # change kwargs in config
      modified_configs = change_config_with_key_value(config, key, value)
      do_logging(
        f'Algo({config.algorithm}): All "{key}" appeared in the following configs will be changed to "{value}": {modified_configs}', 
        backtrack=3, 
        color='cyan'
      )
      # assert modified_configs != [], modified_configs

def model_name_from_kw_string(kw, model_name=''):
  if kw:
    for s in kw:
      key, value = s.split('=', 1)
      if key == 'debug':
        continue
      value = eval_str(value)
      if model_name != '':
        model_name += '-'
      model_name += s
  return model_name

def read_config(algo, env, filename=None):
  configs_dir = get_configs_dir(algo)
  if filename is None:
    filename = get_filename_with_env(env)
  filename = filename + '.yaml'
  path = os.path.join(configs_dir, filename)
  config = load_config(path)

  config = dict2AttrDict(config)

  return config

def load_config_with_algo_env(algo, env, filename=None, to_attrdict=True, return_path=False):
  configs_dir = get_configs_dir(algo)
  if filename is None:
    filename = get_filename_with_env(env)
  filename = filename + '.yaml'
  path = os.path.join(configs_dir, filename)

  config = load_config(path)
  if config is None:
    raise RuntimeError('No configure is loaded')

  suite, name = env.split('-', 1)
  config.env.suite = suite
  config.env.name = name
  config = modify_config(
    config, 
    overwrite_existed_only=True, 
    algorithm=algo, 
    name=algo, 
    info=algo, 
    env_name=env, 
  )

  if to_attrdict:
    config = dict2AttrDict(config)
  if return_path:
    return config, path
  return config


def search_for_all_configs(directory, to_attrdict=True):
  if not os.path.exists(directory):
    return []

  config_files = search_for_all_files(directory, 'config.yaml')
  if config_files == []:
    raise RuntimeError(f'No configure file is found in {directory}')
  configs = [load_config(f, to_attrdict=to_attrdict) for f in config_files]
  if any([c is None for c in configs]):
    raise RuntimeError(f'No configure file is found in {directory}')
  return configs


def search_for_config(directory, to_attrdict=True, check_duplicates=True):
  if isinstance(directory, tuple):
    directory = os.path.join(*directory)

  if not os.path.exists(directory):
    raise ValueError(f'Invalid directory: {directory}')
  
  config_file = search_for_file(directory, 'config.yaml', check_duplicates)
  if config_file is None:
    raise RuntimeError(f'No configure file is found in {directory}')
  config = load_config(config_file, to_attrdict=to_attrdict)
  if config is None:
    raise RuntimeError(f'No configure file is found in {directory}')
  
  return config
