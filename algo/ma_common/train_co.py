import os
import importlib
import ray

from core.elements.builder import ElementsBuilder
from tools.log import do_logging
from core.names import ANCILLARY
from core.utils import configure_gpu, set_seed, save_code_for_seed
from env.utils import divide_env_output
from tools.display import print_dict
from tools.utils import modify_config, flatten_dict
from tools.timer import Timer, timeit, Every
from tools.yaml_op import load_config
from algo.ma_common.run import CurriculumRunner
from algo.ma_common.train import *


@timeit
def ego_optimize(agent, **kwargs):
  agent.train_record(**kwargs)
  train_step = agent.get_train_step()

  return train_step


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
  prepare_buffer = importlib.import_module(f'algo.{algo}.run').prepare_buffer

  while env_step < routine_config.MAX_STEPS:
    env_step = env_run(agents, runner, routine_config, prepare_buffer)
    ego_optimize(agents[0])
    agents[1].buffer.clear()
    time2record = to_record(env_step)

    if time2record:
      eval_and_log(agents, runner, routine_config)


def main(configs, train=train, Runner=CurriculumRunner):
  config = configs[0]
  if config.routine.compute_return_at_once:
    config.buffer.sample_keys += ['advantage', 'v_target']
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
  # Load the opponent configuration from config.opp_config
  opp_config = load_config(config.opp_config)
  opp_config = config.routine.copy()
  agents = [
    build_agent(config, env_stats), 
    build_agent(opp_config, env_stats, rename_model_name=False, save_config=False), 
  ]
  agents = [build_agent(config, env_stats) for config in configs]

  routine_config = config.routine.copy()
  train(
    agents, 
    runner, 
    routine_config, 
  )

  do_logging('Training completed', level='info')
