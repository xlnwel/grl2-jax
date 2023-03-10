from algo.mambpo.train import *


@timeit
def mix_run(agent, model, buffer, model_buffer, routine_config, rng):
    if not model.trainer.is_trust_worthy() \
        or not model_buffer.ready_to_sample():
        return
    def get_agent_states():
        state = agent.get_states()
        # we collect lookahead data into the slow replay
        if isinstance(buffer, DualReplay):
            buffer.set_default_replay(routine_config.lookahead_replay)
        return state
    
    def set_agent_states(states):
        agent.set_states(states)
        if isinstance(buffer, DualReplay):
            buffer.set_default_replay('primal')

    # train lookahead agent
    routine_config = routine_config.copy()
    routine_config.lookahead_rollout = 'uni'
    with TempStore(get_agent_states, set_agent_states):
        run_on_model(
            model, model_buffer, agent, buffer, routine_config, rng)


def train(
    agent, 
    model, 
    runner, 
    buffer, 
    model_buffer, 
    routine_config,
    model_routine_config,
    lka_run_fn=lookahead_run, 
    lka_opt_fn=lookahead_optimize, 
    lka_train_fn=lookahead_train, 
    ego_run_fn=ego_run, 
    ego_opt_fn=ego_optimize, 
    ego_train_fn=ego_train, 
    model_train_fn=model_train
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
    runner.run(MODEL_EVAL_STEPS, agent, None, None, [])
    rng = agent.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, lka_rng = jax.random.split(rng, 2)
        errors = AttrDict()
        env_step = ego_run_fn(
            agent, runner, buffer, model_buffer, routine_config)
        time2record = to_record(env_step)
        
        model_train_fn(model)
        if routine_config.quantify_model_errors and time2record:
            errors.train = quantify_model_errors(
                agent, model, runner.env_config(), MODEL_EVAL_STEPS, [])
        if model is None or (model_routine_config.model_warm_up and env_step < model_routine_config.model_warm_up_steps):
            pass
        else:
            lka_opt_fn(agent)
            # lka_train_fn(
            #     agent, 
            #     model, 
            #     buffer, 
            #     model_buffer, 
            #     routine_config, 
            #     n_runs=routine_config.n_lookahead_steps, 
            #     run_fn=lka_run_fn, 
            #     opt_fn=lka_opt_fn
            # )
            # mix_run(agent, model, buffer, model_buffer, routine_config)
            lka_run_fn(agent, model, buffer, model_buffer, routine_config, lka_rng)
        
        if routine_config.quantify_model_errors and time2record:
            errors.lka = quantify_model_errors(
                agent, model, runner.env_config(), MODEL_EVAL_STEPS, None)

        train_step = ego_opt_fn(agent)
        if routine_config.quantify_model_errors and time2record:
            errors.ego = quantify_model_errors(
                agent, model, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            evaluate(agent, model, runner, env_step, routine_config)
            save(agent, model)
            if routine_config.quantify_model_errors:
                outdir = modelpath2outdir(agent.get_model_path())
                log_model_errors(errors, outdir, env_step)
            log(agent, model, env_step, train_step, errors)

main = partial(main, train=train)
