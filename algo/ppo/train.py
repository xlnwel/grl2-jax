from functools import partial

from core.log import do_logging
from tools.timer import Every
from algo.ma_common.train import *
from algo.ppo.run import prepare_buffer


def train(
  agent, 
  runner: Runner, 
  routine_config, 
):
  do_logging('Training starts...')
  env_step = agent.get_env_step()
  to_record = Every(
    routine_config.LOG_PERIOD, 
    start=env_step, 
    init_next=env_step != 0, 
    final=routine_config.MAX_STEPS
  )
  init_running_stats(agent, runner)

  while env_step < routine_config.MAX_STEPS:
    env_step = env_run(agent, runner, routine_config, prepare_buffer)
    ego_optimize(agent)
    time2record = to_record(env_step)

    if time2record:
      eval_and_log(agent, runner, routine_config)


main = partial(main, train=train, Runner=Runner)
