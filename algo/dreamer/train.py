from functools import partial
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import ModelPath
from tools.display import print_dict, print_dict_info
from tools.store import StateStore
from tools.utils import modify_config
from tools.timer import Every, Timer
from .run import *


def state_constructor_with_sliced_envs(agents, runner):
    agent_states = [a.build_memory() for a in agents]
    env_config = runner.env_config()
    env_config.n_envs //= len(agents)
    runner_states = runner.build_env(env_config)
    return agent_states, runner_states


def state_constructor(agents, runner):
    agent_states = [a.build_memory() for a in agents]
    runner_states = runner.build_env()
    return agent_states, runner_states


def get_states(agents, runner):
    agent_states = [a.get_memory() for a in agents]
    runner_states = runner.get_states()
    return agent_states, runner_states


def set_states(states, agents, runner):
    agent_states, runner_states = states
    assert len(agents) == len(agent_states)
    for a, s in zip(agents, agent_states):
        a.set_memory(s)
    runner.set_states(runner_states)

def train(
    agents,
    model,
    runner, 
    env_buffer,
    model_buffers, 
    routine_config, 
 ):
    do_logging('Training starts...')
    env_step = agents[0].get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    all_aids = list(range(len(agents)))
    steps_per_iter = runner.env.n_envs * routine_config.n_steps

    while env_step < routine_config.MAX_STEPS:

        runner.env_run(
            routine_config.n_steps,
            agents, env_buffer,
            all_aids, False,
            # compute_return=routine_config.compute_return_at_once
        )
        
        if env_buffer.ready_to_sample():
            # train the model
            with Timer('train'):
                model.model_train_record()
    
        env_step += steps_per_iter
        
        print('okkk.')


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

    # load agents
    env_stats = runner.env_stats()
    
    assert len(configs) == env_stats.n_agents, (len(configs), env_stats.n_agents)
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    agents = []
    model_buffers = []
    env_buffer = None
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
        if c.routine.compute_return_at_once:
            c.buffer.sample_keys += ['advantage', 'v_target']
        builder = ElementsBuilder(
            configs[i], 
            env_stats, 
            to_save_code=False, 
            max_steps=config.routine.MAX_STEPS
        )
        elements = builder.build_agent_from_scratch()
        agents.append(elements.agent)
        model_buffers.append(elements.buffer)
    elements = builder.build_agent_from_scratch()
    model = elements.agent
    env_buffer = elements.buffer 
    # TODO

    routine_config = configs[0].routine.copy()
    train(
        agents,
        model,
        runner, 
        env_buffer, 
        model_buffers,
        routine_config
    )

    do_logging('Training completed')
