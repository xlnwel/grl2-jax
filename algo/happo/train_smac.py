from functools import partial

from algo.ma_common.train import *
from algo.lka_common.train import lka_optimize
from algo.happo.train import env_run, eval_ego_and_lka, load_eval_data, eval_policy_distances


def train(
    agent, 
    runner: Runner, 
    routine_config, 
    # env_run=env_run, 
    # ego_optimize=ego_optimize
):
    MODEL_EVAL_STEPS = runner.env.max_episode_steps
    do_logging(f'Model evaluation steps: {MODEL_EVAL_STEPS}')
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    runner.run(
        agent, 
        n_steps=MODEL_EVAL_STEPS, 
        lka_aids=[], 
        collect_data=False
    )
    env_name = runner.env_config().env_name
    eval_data = load_eval_data(filename=env_name)

    while env_step < routine_config.MAX_STEPS:
        env_step = env_run(agent, runner, routine_config, lka_aids=[], name=None)
        lka_optimize(agent)
        train_step = ego_optimize(agent)
        time2record = to_record(env_step)

        if time2record:
            eval_policy_distances(agent, eval_data)
            save(agent, None)
            log(agent, None, env_step, train_step, {})


main = partial(main, train=train)
