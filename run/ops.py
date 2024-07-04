import os
import logging
from datetime import datetime

from core.builder import ElementsBuilder
from tools.log import do_logging
from core.names import PATH_SPLIT
from core.typing import dict2AttrDict
from tools.file import get_configs_dir, get_filename_with_env, \
  search_for_config, search_for_all_configs, load_config_with_algo_env
from tools import pkg
from tools.timer import get_current_datetime
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


def _get_algo_name(algo):
  # shortcuts for distributed algorithms
  algo_mapping = {
    # 'champion_pilot': 'sync-champion_pilot',
  }
  if algo in algo_mapping:
    return algo_mapping[algo]
  return algo


def _get_algo_env_config(cmd_args):
  algos = cmd_args.algorithms
  env = cmd_args.environment
  configs = list(cmd_args.configs)
  if len(algos) < len(configs):
    envs = [env for _ in configs]
  else:
    envs = [env for _ in algos]
  if len(algos) < len(envs):
    assert len(algos) == 1, algos
    algos = [algos[0] for _ in envs]
  assert len(algos) == len(envs), (algos, envs)

  if len(configs) == 0:
    configs = [get_filename_with_env(env) for env in envs]
  elif len(configs) < len(envs):
    configs = [configs[0] for _ in envs]
  assert len(algos) == len(envs) == len(configs), (algos, envs, configs)

  if len(algos) == 1 and cmd_args.n_agents > 1:
    algos = [algos[0] for _ in range(cmd_args.n_agents)]
    envs = [envs[0] for _ in range(cmd_args.n_agents)]
    configs = [configs[0] for _ in range(cmd_args.n_agents)]
  else:
    cmd_args.n_agents = len(algos)
  assert len(algos) == len(envs) == len(configs) == cmd_args.n_agents, (algos, envs, configs, cmd_args.n_agents)
  
  algo_env_config = list(zip(algos, envs, configs))
  
  return algo_env_config


# def _grid_search(config, main, cmd_args):
#   gs = GridSearch(
#     config, 
#     main, 
#     n_trials=cmd_args.trials, 
#     logdir=cmd_args.logdir, 
#     dir_prefix=cmd_args.prefix,
#     separate_process=True, 
#     delay=cmd_args.delay, 
#     multiprocess=cmd_args.multiprocess
#   )

#   processes = []
#   processes += gs(
#     kw_dict={
#       # 'optimizer:lr': np.linspace(1e-4, 1e-3, 2),
#       # 'meta_opt:lr': np.linspace(1e-4, 1e-3, 2),
#       # 'value_coef:default': [.5, 1]
#     }, 
#   )
#   [p.join() for p in processes]


def make_info(config, info, model_name):
  """ 
  info = info if config.info is empty else "{config.info}-{info}"
  model_info = "{info}-{model_name.split('/')[0]}" 
  """
  if info is None:
    info = ''
  if info:
    model_info = f'{info}-{model_name.split(PATH_SPLIT, 1)[0]}'
  else:
    model_info = model_name.split(PATH_SPLIT, 1)[0]
  if config.info and config.info not in info:
    if info:
      info = f'{config.info}-{info}'
    else:
      info = config.info
    model_info = f'{config.info}-{model_info}'
  return info, model_info


def setup_configs(cmd_args, algo_env_config):
  logdir = cmd_args.logdir
  prefix = cmd_args.prefix
  
  if cmd_args.model_name[:4].isdigit():
    date = cmd_args.model_name[:4]
    if len(cmd_args.model_name) == 4:
      raw_model_name = ''
    else:
      assert cmd_args.model_name[4] in ['-', '_', PATH_SPLIT], cmd_args.model_name[4]
      raw_model_name = f'{cmd_args.model_name[5:]}'
  else:
    date = datetime.now().strftime("%m%d")
    raw_model_name = cmd_args.model_name

  model_name_base = model_name_from_kw_string(cmd_args.kwargs, raw_model_name)

  configs = []
  kwidx = cmd_args.kwidx
  if kwidx == []:
    kwidx = list(range(len(algo_env_config)))
  current_time = str(get_current_datetime())
  for i, (algo, env, config) in enumerate(algo_env_config):
    model_name = model_name_base
    do_logging(f'Setup configs for algo({algo}) and env({env})', color='yellow')
    algo = _get_algo_name(algo)
    config = load_config_with_algo_env(algo, env, config, cmd_args.dllib)
    config.dllib = cmd_args.dllib
    if cmd_args.new_kw:
      for s in cmd_args.new_kw:
        key, value = s.split('=', 1)
        config[key] = value
    if i in kwidx:
      change_config_with_kw_string(cmd_args.kwargs, config, i)
    if model_name == '':
      model_name = 'baseline'
    if cmd_args.exploiter:
      model_name = f'{model_name}-exploiter'
      config.exploiter = True

    config.info, config.model_info = make_info(config, cmd_args.info, model_name)
    # model_name = config.model_info
    if not cmd_args.grid_search and not cmd_args.trials > 1:
      model_name = os.path.join(model_name, f'seed={cmd_args.seed}')
    
    dir_prefix = prefix + '-' if prefix else prefix
    env_name = dir_prefix + config.env.env_name
    root_dir = os.path.join(logdir, env_name, f'{config.algorithm}-{cmd_args.dllib}')
    config = modify_config(
      config, 
      max_layer=1, 
      root_dir=root_dir, 
      model_name=os.path.join(date, model_name), 
      algorithm=algo, 
      name=algo, 
      seed=cmd_args.seed
    )
    config.date = date
    config.buffer.root_dir = config.buffer.root_dir.replace('logs', 'data')

    config.launch_time = current_time
    configs.append(config)
  
  if len(configs) < cmd_args.n_agents:
    assert len(configs) == 1, len(configs)
    configs[0]['n_agents'] = cmd_args.n_agents
    configs = [dict2AttrDict(configs[0], to_copy=True) 
      for _ in range(cmd_args.n_agents)]
  elif len(configs) == cmd_args.n_agents:
    configs = [dict2AttrDict(c, to_copy=True) for c in configs]
  else:
    raise NotImplementedError

  for i, c in enumerate(configs):
    modify_config(
      c, 
      aid=i,
      seed=i*100 if cmd_args.seed is None else cmd_args.seed+i*100
    )
  
  return configs


def run_with_configs(cmd_args):
  algo_env_config = _get_algo_env_config(cmd_args)

  configs = setup_configs(cmd_args, algo_env_config)

  main = pkg.import_main(
    cmd_args.train_entry, cmd_args.algorithms[0], 
    dllib=cmd_args.dllib
  )
  # if cmd_args.grid_search or cmd_args.trials > 1:
  #   assert len(configs) == 1, 'No support for multi-agent grid search.'
  #   _grid_search(configs[0], main, cmd_args)
  # else:
  main(configs)
