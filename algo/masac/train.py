import os
from functools import partial
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.typing import AttrDict
from core.utils import configure_gpu, set_seed, save_code_for_seed
from tools.display import print_dict
from tools.store import StateStore
from tools.utils import modify_config
from tools.timer import Every, Timer, timeit
from .run import *
from algo.happo_mb.train import log_model_errors, build_model, \
    prepare_model_errors, log_agent, log_model


def state_constructor_with_sliced_envs(agent, runner):
    agent_states = agent.build_memory()
    env_config = runner.env_config()
    env_config.n_envs //= len(agent)
    runner_states = runner.build_env(env_config)
    return agent_states, runner_states


def state_constructor(agent, runner):
    agent_states = agent.build_memory()
    runner_states = runner.build_env()
    return agent_states, runner_states


def get_states(agent, runner):
    agent_states = agent.get_memory()
    runner_states = runner.get_states()
    return agent_states, runner_states


def set_states(states, agent, runner):
    agent_states, runner_states = states
    agent.set_memory(agent_states)
    runner.set_states(runner_states)


@timeit
def ego_run(agent, runner, routine_config):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with StateStore('real', constructor, get_fn, set_fn):
        runner.run(routine_config.n_steps, agent, [], )

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


@timeit
def ego_optimize(agent):
    agent.train_record()
    train_step = agent.get_train_step()

    return train_step


@timeit
def ego_train(agent, runner, routine_config, run_fn, opt_fn):
    env_step = run_fn(agent, runner, routine_config)
    if agent.buffer.ready_to_sample():
        train_step = opt_fn(agent)
    else:
        train_step = agent.get_train_step()

    return env_step, train_step


@timeit
def evaluate(agent, model, runner, env_step, routine_config):
    if routine_config.EVAL_PERIOD:
        get_fn = partial(get_states, agent=agent, runner=runner)
        set_fn = partial(set_states, agent=agent, runner=runner)
        def constructor():
            env_config = runner.env_config()
            if routine_config.n_eval_envs:
                env_config.n_envs = routine_config.n_eval_envs
            agent_states = agent.build_memory()
            runner_states = runner.build_env()
            return agent_states, runner_states

        with Timer('eval'):
            with StateStore('eval', constructor, get_fn, set_fn):
                eval_scores, eval_epslens, _, video = runner.eval_with_video(
                    agent, record_video=routine_config.RECORD_VIDEO
                )
        
        agent.store(**{
            'eval_score': eval_scores, 
            'eval_epslen': eval_epslens, 
        })
        if model is not None:
            model.store(**{
                'model_eval_score': eval_scores, 
                'model_eval_epslen': eval_epslens, 
            })
        if video is not None:
            agent.video_summary(video, step=env_step, fps=1)


@timeit
def save(agent, model):
    agent.save()
    if model is not None: 
        model.save()


@timeit
def log(agent, model, env_step, train_step, errors):
    error_stats = prepare_model_errors(errors)
    score = log_agent(agent, env_step, train_step, error_stats)
    log_model(model, env_step, score, error_stats)


def train(
    agent, 
    runner, 
    routine_config,
    ego_run_fn=ego_run, 
    ego_opt_fn=ego_optimize, 
    ego_train_fn=ego_train, 
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

    while env_step < routine_config.MAX_STEPS:
        errors = AttrDict()
        env_step, train_step = ego_train_fn(
            agent, runner, routine_config, ego_run_fn, ego_opt_fn)
        time2record = to_record(env_step)
        
        if time2record:
            evaluate(agent, None, runner, env_step, routine_config)
            save(agent, None)
            log(agent, None, env_step, train_step, errors)


@timeit
def build_agent(config, env_stats):
    model_name = config.model_name
    new_model_name = '/'.join([model_name, f'a0'])
    modify_config(
        config, 
        model_name=new_model_name, 
    )
    builder = ElementsBuilder(
        config, 
        env_stats, 
        to_save_code=False, 
        max_steps=config.routine.MAX_STEPS
    )
    elements = builder.build_agent_from_scratch()
    agent = elements.agent

    return agent


def main(configs, train=train):
    config = configs[0]
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
    agent = build_agent(config, env_stats)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    train(
        agent, 
        runner, 
        routine_config,
    )

    do_logging('Training completed')
