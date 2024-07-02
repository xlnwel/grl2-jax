import os, sys
# os.environ["XLA_FLAGS"] = '--xla_dump_to=/tmp/foo'
if sys.platform == "linux" or sys.platform == "linux2":
  pass
elif sys.platform == "darwin":
  os.environ["XLA_FLAGS"] = '--xla_gpu_force_compilation_parallelism=1'
# running in a single cpu
# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
               # "intra_op_parallelism_threads=1")

from datetime import datetime

# try:
#   from tensorflow.python.compiler.mlcompute import mlcompute
#   mlcompute.set_mlc_device(device_name='gpu')
#   print("----------M1----------")
# except:
#   print("----------Not M1-----------")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.log import setup_logging, do_logging
from core.names import PATH_SPLIT
from core.utils import configure_jax_gpu
from tools import pkg
from tools.file import load_config_with_algo_env
from tools.utils import modify_config
from run.args import parse_train_args
from run.grid_search import GridSearch
from run.ops import *
from tools.timer import get_current_datetime


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


def _grid_search(config, main, cmd_args):
  gs = GridSearch(
    config, 
    main, 
    n_trials=cmd_args.trials, 
    logdir=cmd_args.logdir, 
    dir_prefix=cmd_args.prefix,
    separate_process=True, 
    delay=cmd_args.delay, 
    multiprocess=cmd_args.multiprocess
  )

  processes = []
  processes += gs(
    kw_dict={
      # 'optimizer:lr': np.linspace(1e-4, 1e-3, 2),
      # 'meta_opt:lr': np.linspace(1e-4, 1e-3, 2),
      # 'value_coef:default': [.5, 1]
    }, 
  )
  [p.join() for p in processes]


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
    root_dir = os.path.join(logdir, env_name, config.algorithm)
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


def _run_with_configs(cmd_args):
  algo_env_config = _get_algo_env_config(cmd_args)

  configs = setup_configs(cmd_args, algo_env_config)

  main = pkg.import_main(
    cmd_args.train_entry, cmd_args.algorithms[0], 
    dllib=cmd_args.dllib
  )
  if cmd_args.grid_search or cmd_args.trials > 1:
    assert len(configs) == 1, 'No support for multi-agent grid search.'
    _grid_search(configs[0], main, cmd_args)
  else:
    main(configs)


if __name__ == '__main__':
  cmd_args = parse_train_args()
  do_logging(cmd_args, level='info')

  setup_logging(cmd_args.verbose)
  if not (cmd_args.grid_search and cmd_args.multiprocess) and cmd_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = f",".join([f"{gpu}" for gpu in cmd_args.gpu])
  if cmd_args.cpu:
    configure_jax_gpu(None)
  processes = []
  if cmd_args.directory != '':
    configs = [search_for_config(d) for d in cmd_args.directory]
    for config in configs:
      config.cpu_only = cmd_args.cpu
    main = pkg.import_main(cmd_args.train_entry, config=configs[0])
    main(configs)
  else:
    _run_with_configs(cmd_args)
