from tools.store import StateStore
from replay.dual import DualReplay, PRIMAL_REPLAY
from algo.lka_common.train import load_eval_data, eval_and_log
from algo.mambpo_lka.train import *


@timeit
def mix_run(agent, dynamics, routine_config, dynamics_routine_config, rng, name='mix'):
    if dynamics_routine_config.model_warm_up and \
        agent.get_env_step() < dynamics_routine_config.model_warm_up_steps:
        return

    def constructor():
        return agent.build_memory()
    
    def enter_set(states):
        states = agent.set_memory(states)
        # we put the data collected from the dynamics into the secondary replay
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(routine_config.lookahead_replay)
        return states
    
    def exit_set(states):
        states = agent.set_memory(states)
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(PRIMAL_REPLAY)
        return states

    # run (pi^i, x^{-i}) for all i
    routine_config = routine_config.copy()
    routine_config.n_simulated_envs //= dynamics.env_stats.n_agents
    with StateStore(name, constructor, enter_set, exit_set):
        for i in range(dynamics.env_stats.n_agents):
            lka_aids = [j for j in range(dynamics.env_stats.n_agents) if j != i]
            branched_rollout(
                agent, dynamics, routine_config, rng, lka_aids
            )


def train(
    agent, 
    dynamics, 
    runner, 
    routine_config, 
    dynamics_routine_config, 
):
    MODEL_EVAL_STEPS = runner.env.max_episode_steps
    print('Model evaluation steps:', MODEL_EVAL_STEPS)
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    runner.run(agent, n_steps=MODEL_EVAL_STEPS, lka_aids=[], collect_data=False)
    env_name = runner.env_config().env_name
    eval_data = load_eval_data(filename=env_name)
    rng = agent.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, run_rng = jax.random.split(rng, 2)
        # errors = AttrDict()
        env_step = env_run(agent, runner, routine_config, lka_aids=[])
        time2record = to_record(env_step)
        
        dynamics_optimize(dynamics)
        # if routine_config.quantify_dynamics_errors and time2record:
        #     errors.train = quantify_dynamics_errors(
        #         agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if routine_config.lka_test:
            lka_optimize(agent)
            mix_run(
                agent, 
                dynamics, 
                routine_config, 
                dynamics_routine_config, 
                run_rng
            )
        else:
            rngs = jax.random.split(run_rng, 2)
            lka_train(
                agent, 
                dynamics, 
                routine_config, 
                dynamics_routine_config, 
                n_runs=routine_config.n_lka_steps, 
                run_fn=dynamics_run, 
                opt_fn=lka_optimize, 
                rng=rngs[0]
            )
            mix_run(
                agent, 
                dynamics, 
                routine_config, 
                dynamics_routine_config, 
                rngs[1]
            )
        
        # if routine_config.quantify_dynamics_errors and time2record:
        #     errors.lka = quantify_dynamics_errors(
        #         agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, None)

        ego_optimize(agent)
        # if routine_config.quantify_dynamics_errors and time2record:
        #     errors.ego = quantify_dynamics_errors(
        #         agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            eval_and_log(agent, None, None, routine_config, 
                         agent.training_data, eval_data, eval_lka=False)

main = partial(main, train=train)
