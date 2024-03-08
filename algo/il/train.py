import numpy as np

from core.ckpt.pickle import restore
from core.typing import tree_slice
from tools.display import print_dict_info
from tools.timer import Every
from algo.ma_common.train import *


def train(
  agents, 
  runner: Runner, 
  routine_config, 
  # env_run=env_run, 
  # ego_optimize=ego_optimize
):
  MODEL_EVAL_STEPS = runner.env.max_episode_steps
  do_logging(f'Model evaluation steps: {MODEL_EVAL_STEPS}')
  do_logging('Training starts...')
  train_step = agents[0].get_train_step()
  to_record = Every(
    routine_config.LOG_PERIOD, 
    start=train_step, 
    init_next=train_step != 0, 
    final=routine_config.MAX_STEPS
  )
  b = 1000000
  u = runner.env_stats().n_units
  obs_shape = runner.env_stats().obs_shape[0]
  act_dim = runner.env_stats().action_dim[0]
  data = {
    k: np.zeros((b, u, *v)) for k, v in obs_shape.items()
  }
  data['action'] = act_dim
  for k, v in act_dim.items():
    data[k] = np.zeros((b, u, v))
  data['reward'] = np.zeros((b, u))
  data['discount'] = np.zeros((b, u))
  data['reset'] = np.zeros((b, u))
  # data = load_data(filename=routine_config.filename)
  agents[0].buffer.merge(data)

  while train_step < routine_config.MAX_STEPS:
    train_step = ego_optimize(agents)

    if to_record(train_step):
      eval_and_log(agents, runner, routine_config)


def load_data(filename, filedir='/System/Volumes/Data/mnt/公共区/cxw/data', n=None):
  data = restore(filedir=filedir, filename=filename)
  if n is not None:
    maxlen = data['obs'].shape[0]
    indices = np.random.randint(0, maxlen, n)
    data = tree_slice(data, indices)
  print_dict_info(data)
  return data


def main(configs, train=train, Runner=Runner):
  config = configs[0]
  seed = config.get('seed')
  set_seed(seed)

  configure_gpu()
  use_ray = config.env.get('n_runners', 1) > 1
  if use_ray:
    from tools.ray_setup import sigint_shutdown_ray
    from sys import platform
    if platform.startswith('linux'):
      ray.init(num_cpus=config.env.n_runners, num_gpus=1)
    else:
      ray.init(num_cpus=config.env.n_runners)
    sigint_shutdown_ray()

  runner = Runner(config.env)

  env_stats = runner.env_stats()
  env_stats.n_envs = config.env.n_runners * config.env.n_envs
  print_dict(env_stats)

  save_code_for_seed(config)
  agents = [build_agent(config, env_stats) for config in configs]

  routine_config = config.routine.copy()
  train(
    agents, 
    runner, 
    routine_config, 
  )

  do_logging('Training completed')