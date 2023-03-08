from algo.masac.train import *
from algo.mambpo.train import main


@timeit
def ego_run(agent, runner, buffer, model_buffer, routine_config):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with StateStore('real', constructor, get_fn, set_fn):
        runner.run(
            routine_config.n_steps, 
            agent, buffer, 
            model_buffer, 
            None, 
        )

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


def dummy_lookahead_optimize(agent):
    return

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

    while env_step < routine_config.MAX_STEPS:
        errors = AttrDict()
        if model is None or (model_routine_config.model_warm_up and env_step < model_routine_config.model_warm_up_steps):
            pass
        else:
            lka_train_fn(
                agent, 
                model, 
                buffer, 
                model_buffer, 
                routine_config, 
                n_runs=routine_config.n_lookahead_steps, 
                run_fn=lka_run_fn, 
                opt_fn=lka_opt_fn
            )
        
        env_step = ego_run_fn(
            agent, runner, buffer, model_buffer, routine_config)
        time2record = agent.contains_stats('score') \
            and to_record(env_step)
        
        model_train_fn(model)
        if routine_config.quantify_model_errors and time2record:
            errors.train = quantify_model_errors(
                agent, model, runner.env_config(), MODEL_EVAL_STEPS, [])

        if routine_config.quantify_model_errors and time2record:
            errors.lka = quantify_model_errors(
                agent, model, runner.env_config(), MODEL_EVAL_STEPS, None)

        if (not routine_config.use_latest_model) or \
            model is None or (model_routine_config.model_warm_up and env_step < model_routine_config.model_warm_up_steps):
            pass
        else:
            lka_train_fn(
                agent, 
                model, 
                buffer, 
                model_buffer, 
                routine_config, 
                n_runs=routine_config.n_lookahead_steps, 
                run_fn=lka_run_fn, 
                opt_fn=dummy_lookahead_optimize
            )

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
