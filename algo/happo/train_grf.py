from functools import partial

from algo.lka_common.train import *
from algo.happo.train import init_running_stats, env_run


# def record_teammate_ratio(agent):
#     import jax.numpy as jnp
#     from algo.happo.elements.trainer import get_params_and_opt
#     from algo.lka_common.elements.model import pop_lookahead
#     theta = agent.model.theta.copy()
#     theta.policies, is_lookahead = pop_lookahead(theta.policies)
#     assert all([lka == False for lka in is_lookahead]), is_lookahead
#     opt_state = agent.trainer.params.theta.copy()

#     data = agent.training_data
#     teammate_log_ratio = jnp.zeros_like(data.mu_logprob[:, :, :1])

#     for i, aid in enumerate(np.random.permutation(agent.trainer.n_agents)):
#         agent_theta, agent_opt_state = get_params_and_opt(theta, opt_state, aid)
#         _, _, stats = agent.trainer.jit_train(
#             agent_theta, 
#             opt_state=agent_opt_state, 
#             data=agent.training_data, 
#             teammate_log_ratio=teammate_log_ratio, 
#             aid=aid,
#             compute_teammate_log_ratio=True, 
#             return_stats=True
#         )
#         teammate_log_ratio = stats.teammate_log_ratio
#         tm_ratio = np.exp(teammate_log_ratio[0, 0, 0])
#         agent.store(**{f'tm_ratio{i}': tm_ratio})


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
        ego_optimize(agent)
        time2record = to_record(env_step)

        if time2record:
            # record_teammate_ratio(agent)
            eval_and_log(agent, None, None, routine_config, 
                         agent.training_data, eval_data, eval_lka=False)


main = partial(main, train=train)
