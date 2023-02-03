from random import randint
from functools import partial

from core.log import do_logging
from tools.store import StateStore
from tools.timer import Timer
from algo.zero.run import *
from algo.zero.train import main, train, \
    state_constructor, get_states, set_states


def ego_run(agents, runner, buffers, routine_config):
    all_aids = list(range(len(agents)))
    constructor = partial(state_constructor, agents=agents, runner=runner)
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)

    for i, buffer in enumerate(buffers):
        assert buffer.size() == 0, f"buffer {i}: {buffer.size()}"
    with Timer('run'):
        i = randint(0, len(all_aids))   # i in [0, n], n is included to account for the possibility that all agents are looking-ahead
        lka_aids = [aid for aid in all_aids if aid != i]
        with StateStore('real', constructor, get_fn, set_fn):
            runner.run(
                routine_config.n_steps, 
                agents, buffers, 
                lka_aids, all_aids, 
                compute_return=routine_config.compute_return_at_once
            )

    for buffer in buffers:
        assert buffer.ready(), f"buffer {i}: ({buffer.size()}, {len(buffer._queue)})"

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    for agent in agents:
        agent.add_env_step(env_steps_per_run)

    return agents[0].get_env_step()


train = partial(train, ego_run_fn=ego_run)
main = partial(main, train=train)
