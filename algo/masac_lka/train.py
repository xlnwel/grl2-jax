from algo.masac.train import *


def ego_run(agent, runner, buffer, model_buffer, routine_config):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with Timer('run'):
        with StateStore('real', constructor, get_fn, set_fn):
            runner.run(
                routine_config.n_steps, 
                agent, buffer, 
                model_buffer, 
                [], 
            )

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


train = partial(train, ego_run_fn=ego_run)
main = partial(main, train=train)
