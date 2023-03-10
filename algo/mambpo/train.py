from functools import partial

from algo.masac.train import *


def lookahead_optimize(agent):
    return


def update_config(config, model_config):
    config.buffer.primal_replay.model_norm_obs = model_config.buffer.model_norm_obs


train = partial(train, lka_opt_fn=lookahead_optimize)


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
