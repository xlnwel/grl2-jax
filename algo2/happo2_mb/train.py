import os
import collections
from functools import partial
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code_for_seed
from core.typing import get_basic_model_name, modelpath2outdir
from tools.display import print_dict
from tools.plot import prepare_data_for_plotting, lineplot_dataframe
from tools.store import StateStore, TempStore
from tools.utils import modify_config, prefix_name
from tools.timer import Every, Timer, timeit
from .run import *
from algo.ppo.train import state_constructor, get_states, set_states, \
    build_agents, lookahead_optimize, ego_optimize


@timeit
def dynamics_train(dynamics):
    dynamics.train_record()


@timeit
def dynamics_run(agents, dynamics, routine_config, dynamics_routine_config, rng, lka_aids):
    if dynamics_routine_config.model_warm_up and \
        agents[0].get_env_step() < dynamics_routine_config.model_warm_up_steps:
        return

    def get_agent_states():
        state = [a.get_states() for a in agents]
        return state
    
    def set_agent_states(states):
        for a, s in zip(agents, states):
            a.set_states(s)

    # train lookahead agents
    with TempStore(get_agent_states, set_agent_states):
        if not routine_config.switch_model_at_every_step:
            dynamics.model.choose_elite()
        agent_params, dynamics_params = prepare_params(agents, dynamics)
        branched_rollout(
            agents, agent_params, dynamics, dynamics_params, 
            routine_config, rng, lka_aids)


@timeit
def lka_train(agents, dynamics, routine_config, dynamics_routine_config, 
        aids, n_runs, run_fn, opt_fn, rng, lka_aids):
    if not dynamics.trainer.is_trust_worthy() \
        or not dynamics.buffer.ready_to_sample():
        return
    assert n_runs >= 0, n_runs
    for _ in range(n_runs):
        out = run_fn(agents, dynamics, routine_config, dynamics_routine_config, 
            rng, lka_aids)
        if out is not None:
            opt_fn(agents, routine_config, aids)


@timeit
def ego_run(agents, runner, model_buffer, routine_config):
    all_aids = list(range(len(agents)))
    constructor = partial(state_constructor, agents=agents, runner=runner)
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)

    for i, agent in enumerate(agents):
        assert agent.buffer.size() == 0, f"buffer {i}: {agent.buffer.size()}"

    with StateStore('real', constructor, get_fn, set_fn):
        runner.run(
            routine_config.n_steps, 
            agents, 
            model_buffer if routine_config.n_lookahead_steps > 0 else None, 
            all_aids, all_aids, 
            compute_return=routine_config.compute_return_at_once
        )

    for i, agent in enumerate(agents):
        assert agent.buffer.ready(), f"buffer {i}: ({agent.buffer.size()}, {len(agent.buffer._queue)})"

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    for agent in agents:
        agent.add_env_step(env_steps_per_run)

    return agents[0].get_env_step()


@timeit
def ego_train(agents, runner, model_buffer, routine_config, 
        aids, run_fn, opt_fn):
    env_step = run_fn(
        agents, runner, model_buffer, routine_config)
    train_step = opt_fn(agents, routine_config, aids)

    return env_step, train_step


@timeit
def log_dynamics_errors(errors, outdir, env_step):
    if errors:
        data = collections.defaultdict(dict)
        for k1, errs in errors.items():
            for k2, v in errs.items():
                data[k2][k1] = v
        y = 'abs error'
        outdir = '/'.join([outdir, 'errors'])
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        for k, v in data.items():
            filename = f'{k}-{env_step}'
            filepath = '/'.join([outdir, filename])
            with Timer('prepare_data'):
                data[k] = prepare_data_for_plotting(
                    v, y=y, smooth_radius=0, filepath=filepath)
            # lineplot_dataframe(data[k], filename, y=y, outdir=outdir)


@timeit
def evaluate(agents, model, runner, env_step, routine_config):
    if routine_config.EVAL_PERIOD:
        get_fn = partial(get_states, agents=agents, runner=runner)
        set_fn = partial(set_states, agents=agents, runner=runner)
        def constructor():
            env_config = runner.env_config()
            if routine_config.n_eval_envs:
                env_config.n_envs = routine_config.n_eval_envs
            agent_states = [a.build_memory() for a in agents]
            runner_states = runner.build_env()
            return agent_states, runner_states

        with StateStore('eval', constructor, get_fn, set_fn):
            eval_scores, eval_epslens, _, video = runner.eval_with_video(
                agents, record_video=routine_config.RECORD_VIDEO
            )

        agents[0].store(**{
            'eval_score': np.mean(eval_scores), 
            'eval_epslen': np.mean(eval_epslens), 
        })
        if model is not None:
            model.store(**{
                'model_eval_score': eval_scores, 
                'model_eval_epslen': eval_epslens, 
            })
        if video is not None:
            agents[0].video_summary(video, step=env_step, fps=1)


@timeit
def save(agents, model):
    for agent in agents:
        agent.save()
    model.save()


@timeit
def prepare_dynamics_errors(errors):
    if not errors:
        return {}
    error_stats = {}
    for k1, errs in errors.items():
        for k2, v in errs.items():
            error_stats[f'{k1}-{k2}'] = v
    TRAIN = 'train'
    for k1, errs in errors.items():
        for k2 in errs.keys():
            if k1 != TRAIN:
                k1_err = np.mean(error_stats[f'{k1}-{k2}'])
                train_err = np.mean(error_stats[f'{TRAIN}-{k2}'])
                k1_train_err = np.abs(k1_err - train_err)
                error_stats[f'{k1}&{TRAIN}-{k2}'] = k1_train_err
                error_stats[f'norm_{k1}&{TRAIN}-{k2}'] = \
                    k1_train_err / train_err if train_err else k1_train_err
    error_stats = prefix_name(error_stats, 'model_error')

    return error_stats


@timeit
def log_agent(agent, env_step, train_step, error_stats):
    run_time = Timer('ego_run').last()
    train_time = Timer('ego_optimize').last()
    fps = 0 if run_time == 0 else \
        agent.get_env_step_intervals() / run_time
    tps = 0 if train_time == 0 else \
        agent.get_train_step_intervals() / train_time
    
    agent.store(**{
            'stats/train_step': train_step, 
            'time/fps': fps, 
            'time/tps': tps, 
        }, 
        **error_stats, 
        **Timer.top_stats()
    )
    score = agent.get_raw_item('score')
    agent.store(score=score)
    agent.record(step=env_step)
    return score


@timeit
def log_dynamics(model, env_step, score, error_stats):
    if model is None:
        return
    train_step = model.get_train_step()
    train_time = Timer('model_train').last()
    tps = 0 if train_time == 0 else \
        model.get_train_step_intervals() / train_time
    model.store(**{
            'stats/train_step': train_step, 
            'time/tps': tps, 
        }, 
        **error_stats, 
        **Timer.top_stats()
    )
    model.store(model_score=score)
    model.record(step=env_step)


@timeit
def log(agents, model, env_step, train_step, errors):
    error_stats = prepare_dynamics_errors(errors)
    score = log_agent(agents[0], env_step, train_step, error_stats)
    log_dynamics(model, env_step, score, error_stats)

    for agent in agents:
        agent.clear()


def training_aids(all_aids, routine_config):
    aids = np.random.choice(
        all_aids, size=len(all_aids), replace=False, 
        p=routine_config.perm)
    return aids


def train(
    agents, 
    dynamics, 
    runner, 
    routine_config, 
    dynamics_routine_config, 
    aids_fn=training_aids,
    lka_run_fn=dynamics_run, 
    lka_opt_fn=lookahead_optimize, 
    lka_train_fn=lka_train, 
    ego_run_fn=ego_run, 
    ego_opt_fn=ego_optimize, 
    ego_train_fn=ego_train, 
    model_train_fn=dynamics_train
):
    MODEL_EVAL_STEPS = runner.env.max_episode_steps
    print('Model evaluation steps:', MODEL_EVAL_STEPS)
    do_logging('Training starts...')
    env_step = agents[0].get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    all_aids = list(range(len(agents)))
    runner.run(MODEL_EVAL_STEPS, agents, None, [], [])
    rng = dynamics.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, lka_rng = jax.random.split(rng, 2)
        errors = AttrDict()
        aids = aids_fn(all_aids, routine_config)
        time2record = to_record(env_step)
        
        model_train_fn(dynamics)
        if routine_config.quantify_dynamics_errors and time2record:
            errors.train = quantify_dynamics_errors(
                agents, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if dynamics is None or (dynamics_routine_config.model_warm_up and env_step < dynamics_routine_config.model_warm_up_steps):
            pass
        else:
            lka_train_fn(
                agents, 
                dynamics, 
                routine_config, 
                aids=aids, 
                n_runs=routine_config.n_lookahead_steps, 
                run_fn=lka_run_fn, 
                opt_fn=lka_opt_fn, 
                rng=lka_rng
            )
        if routine_config.quantify_dynamics_errors and time2record:
            errors.lka = quantify_dynamics_errors(
                agents, dynamics, runner.env_config(), MODEL_EVAL_STEPS, None)

        env_step, train_step = ego_train_fn(
            agents, 
            runner, 
            dynamics.buffer, 
            routine_config, 
            aids=aids, 
            run_fn=ego_run_fn, 
            opt_fn=ego_opt_fn
        )
        if routine_config.quantify_dynamics_errors and time2record:
            errors.ego = quantify_dynamics_errors(
                agents, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            evaluate(agents, dynamics, runner, env_step, routine_config)
            if routine_config.quantify_dynamics_errors:
                outdir = modelpath2outdir(agents[0].get_model_path())
                log_dynamics_errors(errors, outdir, env_step)
            save(agents, dynamics)
            log(agents, dynamics, env_step, train_step, errors)


@timeit
def build_dynamics(config, model_config, env_stats):
    root_dir = config.root_dir
    model_name = get_basic_model_name(config.model_name)
    seed = config.seed
    new_model_name = '/'.join([model_name, 'dynamics'])
    model_config = modify_config(
        model_config, 
        max_layer=1, 
        aid=0,
        algorithm=config.dynamics_name, 
        name=config.algorithm, 
        info=config.info,
        model_info=config.model_info,
        n_runners=config.env.n_runners, 
        n_envs=config.env.n_envs, 
        root_dir=root_dir, 
        model_name=new_model_name, 
        overwrite_existed_only=True, 
        seed=seed+1000
    )

    builder = ElementsBuilder(
        model_config, 
        env_stats, 
        to_save_code=False, 
        max_steps=config.routine.MAX_STEPS
    )
    elements = builder.build_agent_from_scratch(config=model_config)
    dynamics = elements.agent

    return dynamics


def main(configs, train=train):
    assert len(configs) > 1, len(configs)
    config, model_config = configs[0], configs[-1]
    if config.routine.compute_return_at_once:
        config.buffer.sample_keys += ['advantage', 'v_target']
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
    # assert len(configs) == env_stats.n_agents, (len(configs), env_stats.n_agents)
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    # build agents
    agents = build_agents(config, env_stats)
    # build dynamics
    dynamics = build_dynamics(config, model_config, env_stats)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    dynamics_routine_config = model_config.routine.copy()
    train(
        agents, 
        dynamics, 
        runner, 
        routine_config, 
        dynamics_routine_config
    )

    do_logging('Training completed')
