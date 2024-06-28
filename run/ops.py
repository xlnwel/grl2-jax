import os
import logging

from core.elements.builder import ElementsBuilder
from tools.log import do_logging
from core.names import PATH_SPLIT
from core.typing import dict2AttrDict
from tools.file import get_configs_dir, get_filename_with_env, search_for_config, search_for_all_configs
from tools.utils import eval_str, modify_config
from tools.yaml_op import load_config

logger = logging.getLogger(__name__)


def change_config_with_key_value(config, key, value, prefix=''):
  modified_configs = []
  original_key = key
  if ':' in key:
    keys = key.split(':')
    key = keys[0]
  else:
    keys = None
  if isinstance(value, str) and ':' in value:
    value = [[eval_str(vv) for vv in v.split(',')] for v in value.split(':')]
  elif isinstance(value, str) and ',' in value:
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


def setup_configs(args):
  # load respective config
  if len(args.directory) == 1:
    configs = search_for_all_configs(args.directory[0])
    directories = [args.directory[0] for _ in configs]
  else:
    configs = [search_for_config(d) for d in args.directory]
    directories = args.directory

  # set up env_config
  for d, config in zip(directories, configs):
    if not d.startswith(config.root_dir):
      i = d.find(config.root_dir)
      if i == -1:
        names = d.split(PATH_SPLIT)
        root_dir = os.path.join(n for n in names if n not in config.model_name)
        model_name = os.path.join(n for n in names if n in config.model_name)
        model_name = config.model_name[config.model_name.find(model_name):]
      else:
        root_dir = d[:i] + config.root_dir
        model_name = config.model_name
      do_logging(f'root dir: {root_dir}')
      do_logging(f'model name: {model_name}')
      config = modify_config(
        config, 
        overwrite_existed_only=True, 
        root_dir=root_dir, 
        model_name=model_name
      )
    if args.n_runners:
      config = modify_config(
        config, 
        overwrite_existed_only=True, 
        max_layer=2, 
        n_runners=args.n_runners)
    if args.n_envs:
      config = modify_config(
        config, 
        overwrite_existed_only=True, 
        max_layer=2, 
        n_envs=args.n_envs)
  return configs


def compute_episodes(args):
  n = args.n_episodes
  n = max(args.n_runners * args.n_envs, n)
  return n


def build_agents(configs, env_stats):
  agents = []
  for config in configs:
    builder = ElementsBuilder(config, env_stats)
    elements = builder.build_acting_agent_from_scratch(to_build_for_eval=True)
    agents.append(elements.agent)
  return agents
