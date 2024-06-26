import os
import importlib
import ray

from th.core.typing import get_basic_model_name
from th.core.utils import configure_torch_gpu, set_seed, save_code_for_seed
from tools.log import do_logging
from tools.utils import modify_config
from tools.timer import timeit, Every
from algo.ma_common.train import *
from th.core.elements.builder import ElementsBuilder


@timeit
def build_agent(config, env_stats, aid=0, rename_model_name=True, 
                save_monitor_stats_to_disk=True, save_config=True):
  if rename_model_name:
    model_name = get_basic_model_name(config.model_name)
    new_model_name = os.path.join(model_name, f'a{aid}')
    modify_config(config, model_name=new_model_name)
  builder = ElementsBuilder(
    config, 
    env_stats, 
    to_save_code=False, 
    max_steps=config.routine.MAX_STEPS
  )
  elements = builder.build_agent_from_scratch(
    save_monitor_stats_to_disk=save_monitor_stats_to_disk, 
    save_config=save_config
  )
  agent = elements.agent

  return agent


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
  prepare_buffer = importlib.import_module(f'algo_th.{algo}.run').prepare_buffer

  while env_step < routine_config.MAX_STEPS:
    env_step = env_run(agents, runner, routine_config, prepare_buffer)
    ego_optimize(agents)
    time2record = to_record(env_step)

    if time2record:
      eval_and_log(agents, runner, routine_config)


def main(configs, train=train, Runner=Runner):
  config = configs[0]
  device = configure_torch_gpu(0)
  
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
  do_logging(env_stats, prefix='Env stats', level='info')

  save_code_for_seed(config)
  agents = []
  for aid, config in enumerate(configs):
    config.model.device = device
    config.trainer.device = device
    agents.append(build_agent(config, env_stats, aid=aid))

  routine_config = config.routine.copy()
  train(agents, runner, routine_config)

  do_logging('Training completed', level='info')
