from functools import partial
import jax
import ray

from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code_for_seed
from core.typing import AttrDict, modelpath2outdir
from tools.display import print_dict
from tools.store import StateStore
from tools.timer import Every, timeit
from algo.ma_common.train import state_constructor, get_states, set_states
from algo.lka_common.run import quantify_dynamics_errors
from algo.lka_common.train import dynamics_run, dynamics_optimize, \
    lka_optimize, lka_train, ego_optimize, evaluate, log_dynamics_errors, \
        save, log, build_agent, build_dynamics
from algo.happo.run import prepare_buffer
from algo.happo_mb.run import branched_rollout, Runner


def update_config(config, dynamics_config):
    if config.routine.compute_return_at_once:
        config.buffer.sample_keys += ['advantage', 'v_target']

@timeit
def env_run(agent, runner: Runner, dynamics, routine_config, lka_aids):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with StateStore('real', constructor, get_fn, set_fn):
        env_output = runner.run(routine_config.n_steps, agent, dynamics, lka_aids)
    prepare_buffer(agent, env_output, routine_config.compute_return_at_once)

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


@timeit
def ego_train(agent, runner, dynamics, routine_config, 
        lka_aids, run_fn, opt_fn):
    env_step = run_fn(agent, runner, dynamics, routine_config, lka_aids)
    train_step = opt_fn(agent)

    return env_step, train_step


dynamics_run = partial(dynamics_run, rollout_fn=branched_rollout)


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
    runner.run(MODEL_EVAL_STEPS, agent, None, [], collect_data=False)
    rng = dynamics.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, lka_rng = jax.random.split(rng, 2)
        errors = AttrDict()
        time2record = to_record(env_step)
        
        dynamics_optimize(dynamics)
        if routine_config.quantify_dynamics_errors and time2record:
            errors.train = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        lka_train(
            agent, 
            dynamics, 
            routine_config, 
            dynamics_routine_config, 
            n_runs=routine_config.n_lookahead_steps, 
            rng=lka_rng, 
            lka_aids=[], 
            run_fn=dynamics_run, 
            opt_fn=lka_optimize, 
        )
        if routine_config.quantify_dynamics_errors and time2record:
            errors.lka = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, None)

        env_step, train_step = ego_train(
            agent, 
            runner, 
            dynamics, 
            routine_config, 
            lka_aids=None, 
            run_fn=env_run, 
            opt_fn=ego_optimize
        )
        if routine_config.quantify_dynamics_errors and time2record:
            errors.ego = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            evaluate(agent, dynamics, runner, env_step, routine_config)
            if routine_config.quantify_dynamics_errors:
                outdir = modelpath2outdir(agent.get_model_path())
                log_dynamics_errors(errors, outdir, env_step)
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