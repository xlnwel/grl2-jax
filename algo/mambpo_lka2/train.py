from algo.mambpo_lka.train import *


@timeit
def mix_run(agent, dynamics, routine_config, dynamics_routine_config, rng):
    if dynamics_routine_config.model_warm_up and \
        agent.get_env_step() < dynamics_routine_config.model_warm_up_steps:
        return

    def get_agent_states():
        state = agent.get_states()
        # we collect lookahead data into the slow replay
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(routine_config.lookahead_replay)
        return state
    
    def set_agent_states(states):
        agent.set_states(states)
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(PRIMAL_REPLAY)

    # run (pi^i, x^{-i}) for all i
    routine_config = routine_config.copy()
    routine_config.n_simulated_envs //= dynamics.env_stats.n_agents
    with TempStore(get_agent_states, set_agent_states):
        for i in range(dynamics.env_stats.n_agents):
            lka_aids = [j for j in range(dynamics.env_stats.n_agents) if j != i]
            agent_params, dynamics_params = prepare_params(agent, dynamics)
            branched_rollout(
                agent, agent_params, dynamics, dynamics_params, 
                routine_config, rng, lka_aids
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
    runner.run(MODEL_EVAL_STEPS, agent, [], collect_data=False)
    rng = agent.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, run_rng = jax.random.split(rng, 2)
        errors = AttrDict()
        env_step = env_run(agent, runner, routine_config, lka_aids=[])
        time2record = to_record(env_step)
        
        dynamics_optimize(dynamics)
        if routine_config.quantify_dynamics_errors and time2record:
            errors.train = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

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
                n_runs=routine_config.n_lookahead_steps, 
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
        
        if routine_config.quantify_dynamics_errors and time2record:
            errors.lka = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, None)

        train_step = ego_optimize(agent)
        if routine_config.quantify_dynamics_errors and time2record:
            errors.ego = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            eval_ego_and_lka(agent, runner, routine_config)
            save(agent, dynamics)
            if routine_config.quantify_dynamics_errors:
                outdir = modelpath2outdir(agent.get_model_path())
                log_dynamics_errors(errors, outdir, env_step)
            log(agent, dynamics, env_step, train_step, errors)

main = partial(main, train=train)
