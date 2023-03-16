from algo.ma_common.train import *
from algo.happo.run import prepare_buffer


@timeit
def env_run(agent, runner: Runner, routine_config, lka_aids):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with StateStore('real', constructor, get_fn, set_fn):
        env_output = runner.run(routine_config.n_steps, agent, lka_aids)
    prepare_buffer(agent, env_output, routine_config.compute_return_at_once)

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


train = partial(train, env_run=env_run)
main = partial(main, train=train)
