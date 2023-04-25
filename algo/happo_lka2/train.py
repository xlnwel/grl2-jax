from functools import partial

from algo.lka_common.train import *
from algo.happo.train import init_running_stats, lka_env_run, env_run
from algo.happo_mb.train import get_lka_aids


def train(
    agent, 
    runner: Runner, 
    routine_config, 
    # env_run=env_run, 
    # ego_optimize=ego_optimize
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
    env_name = runner.env_config().env_name
    eval_data = load_eval_data(filename=env_name)

    rollout_type = routine_config.get('rollout_type', 'mix')
    assert rollout_type in ('lka', 'mix'), rollout_type
    n_agents = runner.env_stats().n_agents
    while env_step < routine_config.MAX_STEPS:
        for _ in range(routine_config.n_lka_steps):
            lka_env_run(agent, runner, routine_config, lka_aids=[], store_info=True)
            lka_optimize(agent)
        lka_aids = get_lka_aids(rollout_type, n_agents)
        env_step = env_run(agent, runner, routine_config, 
                           lka_aids=lka_aids, store_info=False)
        ego_optimize(agent)
        time2record = to_record(env_step)

        if time2record:
            eval_and_log(agent, None, runner, routine_config, 
                         agent.training_data, eval_data)


main = partial(main, train=train)
