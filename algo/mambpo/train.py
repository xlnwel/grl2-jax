from functools import partial

from algo.masac.train import *
from algo.mambpo.run import *


def lookahead_optimize(agent):
    return


def update_config(config, model_config):
    config.buffer.primal_replay.model_norm_obs = model_config.buffer.model_norm_obs


train = partial(train, lka_opt_fn=lookahead_optimize)

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
        env_step = ego_run_fn(
            agent, runner, buffer, model_buffer, routine_config)
        train_step = env_step
        time2record = to_record(env_step)
        
        # model_train_fn(model)
        # if routine_config.quantify_model_errors and time2record:
        #     errors.train = quantify_model_errors(
        #         agent, model, runner.env_config(), MODEL_EVAL_STEPS, [])

        # if model is None or (model_routine_config.model_warm_up and env_step < model_routine_config.model_warm_up_steps):
        #     pass
        # else:
        run_on_model(
            model, 
            model_buffer, 
            agent, 
            buffer, 
            routine_config, 
        )

        # train_step = ego_opt_fn(agent)
        # if routine_config.quantify_model_errors and time2record:
        #     errors.ego = quantify_model_errors(
        #         agent, model, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            evaluate(agent, model, runner, env_step, routine_config)
            if routine_config.quantify_model_errors:
                outdir = modelpath2outdir(agent.get_model_path())
                log_model_errors(errors, outdir, env_step)
            save(agent, model)
            log(agent, model, env_step, train_step, errors)


def main(configs, train=train):
    config, model_config = configs[0], configs[-1]
    update_config(config, model_config)
    seed = config.get('seed')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    runner = Runner(config.env)

    # load agent
    env_stats = runner.env_stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    # build agents
    agent, buffer = build_agent(config, env_stats)
    # load model
    model, model_buffer = build_model(config, model_config, env_stats)
    model.change_buffer(buffer)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    model_routine_config = model_config.routine.copy()
    train(
        agent, 
        model, 
        runner, 
        buffer, 
        model_buffer, 
        routine_config,
        model_routine_config
    )

    do_logging('Training completed')
