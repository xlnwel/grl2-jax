import numpy as np

from tools.utils import batch_dicts
from tools.timer import timeit
from algo.lka_common.run import *
from algo.ma_common.run import Runner
from algo.happo.elements.model import LOOKAHEAD
from algo.happo.run import compute_gae


def add_data_to_buffer(
    agent, 
    data, 
    env_output, 
    states, 
    compute_return=True, 
):
    value = agent.compute_value(env_output, states)
    buffer = agent.buffer
    # stack along the time dimension
    data = batch_dicts(data, lambda x: np.stack(x, 1))
    data.value = np.concatenate([data.value, np.expand_dims(value, 1)], 1)
    reset = env_output.reset
    data.state_reset = np.concatenate([data.state_reset, np.expand_dims(reset, 1)], 1)

    if compute_return:
        value = data.value
        if agent.trainer.config.popart:
            value = agent.trainer.popart.denormalize(data.value)
        value, next_value = value[:, :-1], value[:, 1:]
        data.advantage, data.v_target = compute_gae(
            reward=data.reward, 
            discount=data.discount,
            value=value,
            gamma=buffer.config.gamma,
            gae_discount=buffer.config.gamma * buffer.config.lam,
            next_value=next_value, 
            reset=data.reset,
        )
    data[LOOKAHEAD] = True
    buffer.move_to_queue(data)


@timeit
def branched_rollout(agent, dynamics, routine_config, rng, lka_aids):
    env_output, states = initialize_for_dynamics_run(agent, dynamics, routine_config)
    if env_output is None:
        return

    if not routine_config.switch_model_at_every_step:
        dynamics.model.choose_elite()
    agent.model.switch_params(True, lka_aids)
    agent_params, dynamics_params = prepare_params(agent, dynamics)

    # elite_indices = dynamics.model.elite_indices[:dynamics.model.n_elites]
    data, env_output, states = rollout(
        agent.model, agent_params, 
        dynamics.model, dynamics_params, 
        rng, env_output, states, 
        routine_config.n_simulated_steps, 
        # elite_indices
    )
    add_data_to_buffer(agent, data, env_output, states, 
        routine_config.compute_return_at_once)

    agent.model.switch_params(False, lka_aids)
    agent.model.check_params(False)