from random import randint
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import ModelPath
from tools.store import StateStore, TempStore
from tools.utils import modify_config
from tools.timer import Every, Timer
from algo.ppo.train import *
from .run import *


def model_train(model, model_buffer):
    if model_buffer.ready_to_sample():
        model.train_record()


def lookahead_run(agents, model, buffers, model_buffer, routine_config):
    def get_agent_states():
        state = [a.get_states() for a in agents]
        return state
    
    def set_agent_states(states):
        for a, s in zip(agents, states):
            a.set_states(s)

    # train lookahead agents
    with Timer('lookahead_run'):
        with TempStore(get_agent_states, set_agent_states):
            run_on_model(
                model, model_buffer, agents, buffers, routine_config)


def lookahead_train(agents, model, buffers, model_buffer, routine_config, 
        n_runs, run_fn, opt_fn):
    if not model_buffer.ready_to_sample():
        return
    if n_runs > 0:
        for _ in range(n_runs):
            run_fn(agents, model, buffers, model_buffer, routine_config)
            opt_fn(agents, routine_config)


def ego_run(agents, runner, buffers, model_buffer, routine_config):
    all_aids = list(range(len(agents)))
    constructor = partial(state_constructor, agents=agents, runner=runner)
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)

    for i, buffer in enumerate(buffers):
        assert buffer.size() == 0, f"buffer {i}: {buffer.size()}"

    with Timer('run'):
        i = randint(0, len(all_aids))   # i in [0, n], n is included to account for the possibility that all agents are looking-ahead
        lka_aids = [aid for aid in all_aids if aid != i]
        with StateStore('real', constructor, get_fn, set_fn):
            runner.run(
                routine_config.n_steps, 
                agents, buffers, 
                model_buffer, 
                lka_aids, all_aids, 
                compute_return=routine_config.compute_return_at_once
            )

    for i, buffer in enumerate(buffers):
        assert buffer.ready(), f"buffer {i}: ({buffer.size()}, {len(buffer._queue)})"

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    for agent in agents:
        agent.add_env_step(env_steps_per_run)

    return agents[0].get_env_step()


def ego_train(agents, runner, buffers, model_buffer, routine_config, 
        n_runs, run_fn, opt_fn):
    if n_runs > 0:
        for _ in range(n_runs):
            env_step = run_fn(
                agents, runner, buffers, model_buffer, routine_config)
            train_step = opt_fn(agents, routine_config)
    else:
        env_step = agents[0].get_env_step()
        train_step = agents[0].get_train_step()

    return env_step, train_step


def eval_and_log(agents, model, runner, env_step, routine_config):
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)
    def constructor():
        env_config = runner.env_config()
        if routine_config.n_eval_envs:
            env_config.n_envs = routine_config.n_eval_envs
        agent_states = [a.build_memory() for a in agents]
        runner_states = runner.build_env()
        return agent_states, runner_states

    with Timer('eval'):
        with StateStore('eval', constructor, get_fn, set_fn):
            scores, epslens, _, video = runner.eval_with_video(
                agents, record_video=routine_config.RECORD_VIDEO
            )
    agents[0].store(**{
        'eval_score': np.mean(scores), 
        'eval_epslen': np.mean(epslens), 
    })

    with Timer('save'):
        for agent in agents:
            agent.save()
        model.save()
    with Timer('log'):
        if video is not None:
            agents[0].video_summary(video, step=env_step, fps=1)
        agents[0].record(step=env_step)
        for i in range(1, len(agents)):
            agents[i].clear()
        model.record(step=env_step)


def train(
    agents, 
    model, 
    runner, 
    buffers, 
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
    env_step = agents[0].get_env_step()
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
            agents, model, 
            buffers, model_buffer, 
            routine_config, 
            n_runs=routine_config.n_lookahead_steps, 
            run_fn=lka_run_fn, 
            opt_fn=lka_opt_fn
        )
        env_step, _ = ego_train_fn(
            agents, 
            runner, 
            buffers, 
            model_buffer, 
            routine_config, 
            n_runs=1, 
            run_fn=ego_run_fn, 
            opt_fn=ego_opt_fn
        )

        time2record = agents[0].contains_stats('score') \
            and to_record(env_step)
        if time2record:
            eval_and_log(agents, model, runner, env_step, routine_config)


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

    configs, model_config = configs[:-1], configs[-1]
    # load agents
    env_stats = runner.env_stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    agents = []
    buffers = []
    root_dir = config.root_dir
    model_name = config.model_name
    for i, c in enumerate(configs):
        assert c.aid == i, (c.aid, i)
        if model_name.endswith(f'a{i}'):
            new_model_name = model_name
        else:
            new_model_name = '/'.join([model_name, f'a{i}'])
        modify_config(
            configs[i], 
            model_name=new_model_name, 
        )
        builder = ElementsBuilder(
            configs[i], 
            env_stats, 
            to_save_code=False, 
            max_steps=config.routine.MAX_STEPS
        )
        elements = builder.build_agent_from_scratch()
        agents.append(elements.agent)
        buffers.append(elements.buffer)
    save_code(ModelPath(root_dir, model_name))

    # load model
    new_model_name = '/'.join([model_name, 'model'])
    model_config = modify_config(
        model_config, 
        max_layer=1, 
        aid=0,
        algorithm=configs[0].dynamics_name, 
        n_runners=configs[0].env.n_runners, 
        n_envs=configs[0].env.n_envs, 
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

    routine_config = configs[0].routine.copy()
    if routine_config.perm is None:
        routine_config.perm = list(np.ones(len(agents)) / len(agents))
    train(
        agents, 
        model, 
        runner, 
        buffers, 
        model_buffer, 
        routine_config
    )

    do_logging('Training completed')
