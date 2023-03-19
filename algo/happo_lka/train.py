from functools import partial
import jax
import ray

from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code_for_seed
from core.typing import AttrDict
from tools.display import print_dict
from tools.store import StateStore
from tools.timer import Every, timeit
from algo.ma_common.train import state_constructor, get_states, set_states, \
    ego_optimize, build_agent, save, log, evaluate
from algo.lka_common.train import dynamics_run, lka_optimize, lka_train
from algo.happo.run import prepare_buffer
from algo.happo_lka.run import Runner


def update_config(config, dynamics_config):
    if config.routine.compute_return_at_once:
        config.buffer.sample_keys += ['advantage', 'v_target']


@timeit
def env_run(agent, runner: Runner, routine_config, lka_aids):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with StateStore('real', constructor, get_fn, set_fn):
        env_output = runner.run(routine_config.n_steps, agent, lka_aids)
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


@timeit
def eval_ego_and_lka(agent, dynamics, runner, routine_config, dynamics_routine_config, rng):
    ego_score, _, _ = evaluate(agent, runner, routine_config)
    env_run(agent, runner, routine_config, lka_aids=[])
    lka_optimize(agent)
    lka_score, _, _ = evaluate(agent, runner, routine_config, None)
    agent.trainer.sync_lookahead_params()
    agent.store(
        ego_score=ego_score, 
        lka_score=lka_score, 
        lka_ego_score_diff=[lka - ego for lka, ego in zip(lka_score, ego_score)]
    )


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
    rng = agent.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, lka_rng = jax.random.split(rng, 2)
        time2record = to_record(env_step)
        
        env_run(agent, runner, routine_config, lka_aids=[])
        lka_optimize(agent)
        env_step = env_run(agent, runner, routine_config, lka_aids=None)
        train_step = ego_optimize(agent)

        if time2record:
            eval_ego_and_lka(agent, dynamics, runner, 
                routine_config, dynamics_routine_config, lka_rng)
            save(agent, dynamics)
            log(agent, dynamics, env_step, train_step, AttrDict())


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
    dynamics = None
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
