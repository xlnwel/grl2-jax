from functools import partial

from algo.lka_common.train import *
from algo.happo.run import prepare_buffer


@timeit
def lka_env_run(agent, runner: Runner, routine_config, name='lka', **kwargs):
    env_output = runner.run(
        agent, 
        n_steps=routine_config.n_steps, 
        name=name, 
        **kwargs
    )
    prepare_buffer(agent, env_output, routine_config.compute_return_at_once)


@timeit
def env_run(agent, runner: Runner, routine_config, name='real', **kwargs):
    env_output = runner.run(
        agent, 
        n_steps=routine_config.n_steps, 
        name=name, 
        **kwargs
    )
    prepare_buffer(agent, env_output, routine_config.compute_return_at_once)

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


def init_running_stats(agent, runner: Runner, dynamics=None, n_steps=None):
    if n_steps is None:
        n_steps = runner.env.max_episode_steps
    do_logging(f'Pre-training running steps: {n_steps}')

    runner.run(
        agent, 
        n_steps=n_steps, 
        lka_aids=[], 
        dynamics=dynamics, 
        collect_data=agent.actor.is_obs_or_reward_normalized
    )
    if agent.actor.is_obs_or_reward_normalized:
        data = agent.buffer.get_data()
        agent.actor.update_reward_rms(data.reward, data.discount)
        agent.actor.update_obs_rms(data)
        agent.actor.reset_reward_rms_return()
    agent.actor.print_rms()


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

    while env_step < routine_config.MAX_STEPS:
        env_step = env_run(agent, runner, routine_config, lka_aids=[], store_info=True)
        if not getattr(routine_config, "concise_mode", False):
            lka_optimize(agent)
        ego_optimize(agent)
        time2record = to_record(env_step)

        if time2record:
            eval_and_log(agent, None, runner, routine_config, 
                         agent.training_data, eval_data, eval_lka=False)


main = partial(main, train=train)
