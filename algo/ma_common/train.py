import os
import importlib
import ray

from core.builder import ElementsBuilder
from core.names import ANCILLARY, DL_LIB
from core.typing import get_basic_model_name
from core.utils import configure_gpu, set_seed, save_code_for_seed
from envs.utils import divide_env_output
from tools.log import do_logging
from tools.utils import modify_config, flatten_dict
from tools.timer import Timer, timeit, Every
from tools.pkg import get_package
from algo.ma_common.run import Runner


def init_running_stats(agents, runner: Runner, n_steps=None):
  if n_steps is None:
    n_steps = min(100, runner.env.max_episode_steps)
  do_logging(f'Pre-training running steps: {n_steps}', level='info')

  if agents[0].actor.is_obs_or_reward_normalized:
    runner.run(agents, n_steps=n_steps)
    for agent in agents:
      data = agent.buffer.get_data()
      agent.actor.update_reward_rms(data.reward, data.discount)
      agent.actor.update_obs_rms(
        data, mask=data.sample_mask, feature_mask=agent.env_stats.feature_mask)
      agent.actor.reset_reward_rms_return()
  for agent in agents:
    agent.actor.print_rms()


@timeit
def env_run(agents, runner: Runner, routine_config, prepare_buffer=None, name='real'):
  env_output = runner.run(
    agents, 
    n_steps=routine_config.n_steps, 
    name=name, 
    store_info=True
  )
  if prepare_buffer is not None:
    agent_env_outputs = divide_env_output(env_output)
    for agent, eo in zip(agents, agent_env_outputs):
      prepare_buffer(agent, eo, routine_config.compute_return_at_once)

  env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
  for agent in agents:
    agent.add_env_step(env_steps_per_run)

  return agent.get_env_step()


@timeit
def ego_optimize(agents, **kwargs):
  for agent in agents:
    agent.train_record(**kwargs)
    train_step = agent.get_train_step()

  return train_step


@timeit
def ego_train(agent, runner, routine_config, train_aids, lka_aids, run_fn, opt_fn):
  env_step = run_fn(agent, runner, routine_config, lka_aids)
  train_step = opt_fn(agent, aids=train_aids)

  return env_step, train_step


@timeit
def eval_and_log(agents, runner: Runner, routine_config, record_video=False, name='eval'):
  if routine_config.EVAL_PERIOD:
    scores, epslens, _, video = runner.eval_with_video(
      agents, 
      n_envs=routine_config.n_eval_envs, 
      record_video=record_video, 
      name=name
    )
    for agent in agents:
      agent.store(**{
        'eval_score': scores, 
        'eval_epslen': epslens, 
      })
      if video is not None:
        agent.video_summary(video, step=agent.get_env_step(), fps=1)
  for agent in agents:
    save(agent)
    log(agent)


@timeit
def save(agent):
  agent.save()


@timeit
def log_agent(agent, env_step, train_step, error_stats={}):
  run_time = Timer('env_run').last()
  train_time = Timer('ego_optimize').last()
  fps = 0 if run_time == 0 else agent.get_env_step_intervals() / run_time
  tps = 0 if train_time == 0 else agent.get_train_step_intervals() / train_time
  rms = agent.actor.get_auxiliary_stats()
  rms_dict = {}
  if rms is not None:
    if isinstance(rms[0], list):
      for i, v in enumerate(rms[0]):
        rms_dict[f'{ANCILLARY}/obs{i}'] = v
    else:
      rms_dict[f'{ANCILLARY}/obs'] = rms[0]
    rms_dict[f'{ANCILLARY}/reward'] = rms[-1]
    rms_dict = flatten_dict(rms_dict)

  agent.store(**{
      'stats/train_step': train_step, 
      'time/fps': fps, 
      'time/tps': tps, 
    }, 
    **rms_dict, 
    **error_stats, 
    **Timer.top_stats()
  )
  score = agent.get_raw_item('score')
  agent.store(score=score)
  agent.record(step=env_step)
  return score


@timeit
def log(agent):
  env_step = agent.get_env_step()
  train_step = agent.get_train_step()
  log_agent(agent, env_step, train_step)


@timeit
def build_agent(config, env_stats, aid=0, rename_model_name=True, 
                save_monitor_stats_to_disk=True, save_config=True):
  if rename_model_name:
    model_name = get_basic_model_name(config.model_name)
    new_model_name = os.path.join(model_name, f'a{aid}')
    modify_config(config, model_name=new_model_name)
  modify_config(config, aid=aid, overwrite_existed_only=True)
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
  root_dir = DL_LIB.TORCH if agents[0].dllib == DL_LIB.TORCH else None
  pkg = get_package(root_dir, 'algo', algo)
  prepare_buffer = importlib.import_module(f'{pkg}.run').prepare_buffer

  while env_step < routine_config.MAX_STEPS:
    env_step = env_run(agents, runner, routine_config, prepare_buffer)
    ego_optimize(agents)
    time2record = to_record(env_step)

    if time2record:
      eval_and_log(agents, runner, routine_config)


def main(configs, train=train, Runner=Runner):
  configs = configure_gpu(configs)
  config = configs[0]
  if config.routine.compute_return_at_once:
    config.buffer.sample_keys += ['advantage', 'v_target']
  seed = config.get('seed')
  set_seed(seed, config.dllib)

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
