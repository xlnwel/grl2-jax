from functools import partial
import jax
import ray

from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code_for_seed
from core.typing import AttrDict, modelpath2outdir
from tools.display import print_dict
from tools.timer import Every
from algo.ma_common.run import Runner
from algo.ma_common.train import env_run, ego_optimize, build_agent, save, log
from algo.lka_common.run import quantify_dynamics_errors
from algo.lka_common.train import dynamics_run, dynamics_optimize, \
    build_dynamics, load_eval_data, eval_and_log
from algo.mambpo.run import branched_rollout


dynamics_run = partial(dynamics_run, rollout_fn=branched_rollout)


def update_config(config, dynamics_config):
    config.buffer.primal_replay.model_norm_obs = dynamics_config.buffer.model_norm_obs


def train(
    agent, 
    dynamics, 
    runner: Runner, 
    routine_config, 
    dynamics_routine_config, 
    # dynamics_optimize=dynamics_optimize, 
    # dynamics_run=dynamics_run, 
    # env_run=env_run, 
    # ego_optimize=ego_optimize, 
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
    runner.run(agent, n_steps=MODEL_EVAL_STEPS, lka_aids=[], collect_data=False)
    env_name = runner.env_config().env_name
    eval_data = load_eval_data(filename=env_name)
    rng = agent.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, run_rng = jax.random.split(rng, 2)
        env_step = env_run(agent, runner, routine_config, lka_aids=[])
        time2record = to_record(env_step)
        
        if dynamics_routine_config.model_warm_up and \
            env_step < dynamics_routine_config.model_warm_up_steps:
            dynamics_optimize(dynamics, warm_up_stage=True)
        else:
            dynamics_optimize(dynamics)
        dynamics_run(
            agent, dynamics, 
            routine_config, 
            dynamics_routine_config, 
            run_rng, 
            lka_aids=[]
        )
        # if routine_config.quantify_dynamics_errors and time2record:
        #     errors.train = quantify_dynamics_errors(
        #         agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])
        
        if dynamics_routine_config.model_warm_up and \
            env_step < dynamics_routine_config.model_warm_up_steps:
            ego_optimize(agent, warm_up_stage=True)
        else:
            ego_optimize(agent)

        # if routine_config.quantify_dynamics_errors and time2record:
        #     errors.ego = quantify_dynamics_errors(
        #         agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            eval_and_log(agent, None, None, routine_config, 
                         agent.training_data, eval_data, eval_lka=False)


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
    dynamics.change_buffer(agent.buffer)
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
