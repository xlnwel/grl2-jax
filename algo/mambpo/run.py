from tools.timer import timeit
from tools.utils import yield_from_dict
from algo.lka_common.run import *


def add_data_to_buffer(agent, data):
    for step_data in data:
        for d in yield_from_dict(step_data):
            agent.buffer.merge(d) 

@timeit
def branched_rollout(agent, agent_params, dynamics, dynamics_params, 
                    routine_config, rng, lka_aids):
    env_output = initialize_for_dynamics_run(agent, dynamics, routine_config)
    if env_output is None:
        return

    if not routine_config.switch_model_at_every_step:
        dynamics.model.choose_elite()
    agent.model.switch_params(True, lka_aids)

    # elite_indices = dynamics.model.elite_indices[:dynamics.model.n_elites]
    data, env_output = rollout(
        agent.model, agent_params, 
        dynamics.model, dynamics_params, 
        rng, env_output, routine_config.n_simulated_steps, 
        # elite_indices
    )
    add_data_to_buffer(agent, data)

    agent.model.switch_params(False, lka_aids)
    agent.model.check_params(False)