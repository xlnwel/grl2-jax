from functools import partial

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
  env_step = agents[0].get_env_step()
  to_record = Every(
    routine_config.LOG_PERIOD, 
    start=env_step, 
    init_next=env_step != 0, 
    final=routine_config.MAX_STEPS
  )

  while env_step < routine_config.MAX_STEPS:
    env_step = env_run(agents, runner, routine_config)
    ego_optimize(agents)
    time2record = to_record(env_step)

    if time2record:
      eval_and_log(agents, runner, routine_config)


main = partial(main, train=train)
