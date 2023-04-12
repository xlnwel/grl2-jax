from functools import partial
import jax
import ray

from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code_for_seed
from core.typing import AttrDict
from tools.display import print_dict
from tools.timer import Every
from algo.ma_common.train import ego_optimize, build_agent, save, log
from algo.lka_common.train import dynamics_run, dynamics_optimize, \
    build_dynamics, lka_optimize, lka_train
from algo.happo.train import load_eval_data, eval_policy_distances
from algo.happo_mb.run import Runner
from algo.happo_mb.train import update_config, env_run, ego_train, dynamics_run


env_run = partial(env_run, name=None)


def train(
    agent, 
    dynamics, 
    runner: Runner, 
    routine_config, 
    dynamics_routine_config, 
    # dynamics_optimize=dynamics_optimize, 
    # dynamics_run=dynamics_run, 
    # lka_optimize=lka_optimize, 
    # lka_train=lka_train, 
    # env_run=env_run, 
    # ego_optimize=ego_optimize, 
    # ego_train=ego_train, 
):
    MODEL_EVAL_STEPS = runner.env.max_episode_steps
    do_logging(f'Model evaluation steps: {MODEL_EVAL_STEPS}')
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    runner.run(
        agent, 
        n_steps=MODEL_EVAL_STEPS, 
        lka_aids=[], 
        collect_data=False
    )
    env_name = runner.env_config().env_name
    eval_data = load_eval_data(filename=env_name)
    rng = dynamics.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, lka_rng = jax.random.split(rng, 2)
        errors = AttrDict()
        time2record = to_record(env_step)

        dynamics_optimize(dynamics)
        lka_train(
            agent, 
            dynamics, 
            routine_config, 
            dynamics_routine_config, 
            n_runs=routine_config.n_lka_steps, 
            rng=lka_rng, 
            lka_aids=[], 
            run_fn=dynamics_run, 
            opt_fn=lka_optimize, 
        )
        env_step, train_step = ego_train(
            agent, 
            runner, 
            dynamics, 
            routine_config, 
            lka_aids=None, 
            run_fn=env_run, 
            opt_fn=ego_optimize
        )

        if time2record:
            eval_policy_distances(agent, eval_data)
            save(agent, dynamics)
            log(agent, dynamics, env_step, train_step, errors)


def main(configs, train=train):
    assert len(configs) > 1, len(configs)
    config, dynamics_config = configs[0], configs[-1]
    update_config(config, dynamics_config)
    seed = config.get('seed')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    runner = Runner(config.env)

    env_stats = runner.env_stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    # build agent
    agent = build_agent(config, env_stats)
    # build dynamics
    dynamics = build_dynamics(config, dynamics_config, env_stats)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    dynamics_routine_config = dynamics_config.routine.copy()
    train(
        agent, 
        dynamics, 
        runner, 
        routine_config, 
        dynamics_routine_config
    )

    do_logging('Training completed')
