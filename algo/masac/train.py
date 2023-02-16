from functools import partial
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import ModelPath
from tools.display import print_dict
from tools.store import StateStore, TempStore
from tools.utils import modify_config
from tools.timer import Every, Timer
from .run import *


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


def model_train(model, model_buffer):
    if model_buffer.ready_to_sample():
        with Timer('model_train'):
            model.train_record()


def lookahead_run(agent, model, buffer, model_buffer, routine_config):
    def get_agent_states():
        state = agent.get_states()
        return state
    
    def set_agent_states(states):
        agent.set_states(states)

    # train lookahead agent
    with Timer('lookahead_run'):
        with TempStore(get_agent_states, set_agent_states):
            run_on_model(
                model, model_buffer, agent, buffer, routine_config)


def lookahead_optimize(agent):
    agent.lookahead_train()


def lookahead_train(agent, model, buffer, model_buffer, routine_config, 
        n_runs, run_fn, opt_fn):
    if not model_buffer.ready_to_sample():
        return
    assert n_runs >= 0, n_runs
    old_buffer = agent.change_buffer(buffer)
    for _ in range(n_runs):
        run_fn(agent, model, buffer, model_buffer, routine_config)
        opt_fn(agent)
    agent.change_buffer(old_buffer)


def ego_run(agent, runner, buffer, model_buffer, routine_config):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with Timer('run'):
        with StateStore('real', constructor, get_fn, set_fn):
            runner.run(
                routine_config.n_steps, 
                agent, buffer, 
                model_buffer if routine_config.n_lookahead_steps > 0 else None, 
                [], 
            )

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


def ego_optimize(agent):
    agent.train_record()
    train_step = agent.get_train_step()

    return train_step

def ego_train(agent, runner, buffer, model_buffer, routine_config, 
        run_fn, opt_fn):
    env_step = run_fn(
        agent, runner, buffer, model_buffer, routine_config)
    if buffer.ready_to_sample():
        train_step = opt_fn(agent)
    else:
        train_step = agent.get_train_step()

    return env_step, train_step


def eval_and_log(agent, model, runner, env_step, train_step, routine_config):
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
            scores, epslens, _, video = runner.eval_with_video(
                agent, record_video=routine_config.RECORD_VIDEO
            )
    agent.store(**{
        'eval_score': np.mean(scores), 
        'eval_epslen': np.mean(epslens), 
    })

    with Timer('save'):
        agent.save()
        model.save()

    with Timer('log'):
        if video is not None:
            agent.video_summary(video, step=env_step, fps=1)
        run_time = Timer('run').last()
        train_time = Timer('train').last()
        if run_time == 0:
            fps = 0
        else:
            fps = agent.get_env_step_intervals() / Timer('run').last()
        if train_time == 0:
            tps = 0
        else:
            tps = agent.get_train_step_intervals() / Timer('train').last()
        agent.store(**{
                'stats/train_step': train_step, 
                'time/fps': fps, 
                'time/tps': tps, 
            }, 
            **Timer.all_stats()
        )
        agent.record(step=env_step)
        agent.clear()

        train_step = model.get_train_step()
        model_train_duration = Timer('model_train').last()
        if model_train_duration == 0:
            tps = 0
        else:
            tps = model.get_train_step_intervals() / model_train_duration
        model.store(**{
                'stats/train_step': train_step, 
                'time/tps': tps, 
            }, 
            **Timer.all_stats()
        )
        model.record(step=env_step)


def train(
    agent, 
    model, 
    runner, 
    buffer, 
    lka_buffer, 
    model_buffer, 
    routine_config, 
    lka_run_fn=lookahead_run, 
    lka_opt_fn=lookahead_optimize, 
    lka_train_fn=lookahead_train, 
    ego_run_fn=ego_run, 
    ego_opt_fn=ego_optimize, 
    ego_train_fn=ego_train, 
    model_train_fn=model_train
):
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )

    while env_step < routine_config.MAX_STEPS:        
        model_train_fn(
            model, 
            model_buffer
        )

        lka_train_fn(
            agent, 
            model, 
            lka_buffer, 
            model_buffer, 
            routine_config, 
            n_runs=routine_config.n_lookahead_steps, 
            run_fn=lka_run_fn, 
            opt_fn=lka_opt_fn
        )
        env_step, train_step = ego_train_fn(
            agent, 
            runner, 
            buffer, 
            model_buffer, 
            routine_config, 
            run_fn=ego_run_fn, 
            opt_fn=ego_opt_fn
        )

        time2record = agent.contains_stats('score') \
            and to_record(env_step)
        if time2record:
            eval_and_log(
                agent, model, runner, env_step, train_step, routine_config)


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

    config, model_config = configs[0], configs[-1]
    # load agent
    env_stats = runner.env_stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    root_dir = config.root_dir
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
    buffer = elements.buffer
    lka_buffer = builder.build_buffer(elements.model, env_stats=env_stats)
    if seed == 0:
        save_code(ModelPath(root_dir, model_name))

    # load model
    new_model_name = '/'.join([model_name, 'model'])
    model_config = modify_config(
        model_config, 
        max_layer=1, 
        aid=0,
        algorithm=config.dynamics_name, 
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
    model = elements.agent
    model_buffer = elements.buffer

    routine_config = config.routine.copy()
    train(
        agent, 
        model, 
        runner, 
        buffer, 
        lka_buffer, 
        model_buffer, 
        routine_config
    )

    do_logging('Training completed')
