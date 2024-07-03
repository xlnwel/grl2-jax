import importlib
import ray

from th.core.utils import configure_torch_gpu, set_seed, save_code_for_seed
from tools.log import do_logging
from tools.utils import modify_config
from tools.timer import Every
from algo.ma_common.train import *


def train(
  agents, 
  runner: Runner, 
  routine_config, 
):
  do_logging('Training starts...', level='info')
  env_step = agents[0].get_env_step()
  to_record = Every(
    routine_config.LOG_PERIOD, 
    start=env_step, 
    init_next=env_step != 0, 
    final=routine_config.MAX_STEPS
  )
  init_running_stats(agents, runner, n_steps=routine_config.n_steps)
  algo = agents[0].name
  prepare_buffer = importlib.import_module(f'th.algo.{algo}.run').prepare_buffer

  while env_step < routine_config.MAX_STEPS:
    env_step = env_run(agents, runner, routine_config, prepare_buffer)
    ego_optimize(agents)
    time2record = to_record(env_step)

    if time2record:
      eval_and_log(agents, runner, routine_config)


def main(configs, train=train, Runner=Runner):
  config = configs[0]
  device = configure_torch_gpu(0)
  config = modify_config(config, device=device)
  if config.routine.compute_return_at_once:
    config.buffer.sample_keys += ['advantage', 'v_target']
  seed = config.get('seed')
  set_seed(seed)

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
  do_logging(env_stats, prefix='Env stats')

  save_code_for_seed(config)
  n_agents = max(env_stats.n_agents, len(configs))
  if len(configs) < n_agents:
    agents = [build_agent(config, env_stats, aid=aid) 
              for aid in range(n_agents)]
  else:
    agents = [build_agent(config, env_stats, aid=aid) 
              for aid, config in enumerate(configs)]

  routine_config = config.routine.copy()
  train(agents, runner, routine_config)

  do_logging('Training completed', level='info')
