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
    agent,
    model,
    runner,
    buffer,
    routine_config,
 ):
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    all_aids = list(range(agent.env_stats.n_agents))
    steps_per_iter = runner.env.n_envs * routine_config.n_steps

    while env_step < routine_config.MAX_STEPS:

        # Real world interaction
        runner.env_run(
            routine_config.n_steps,
            agent, buffer,
            all_aids, False,
            # compute_return=routine_config.compute_return_at_once
        )

        # Buffer sampling to train
        if buffer.ready_to_sample():
            # train the model
            with Timer('train'):
                model.model_train_record()

        sample_keys = ['state']
        obs = buffer.sample_from_recency(
            batch_size=routine_config.n_envs, 
            sample_keys=sample_keys, 
            sample_size=1, 
            squeeze=True, 
            n=routine_config.n_recent_trajectories
        )
        rollout_data = model.model_rollout(obs.state, routine_config.rollout_length)

        model.train_record(data=rollout_data)

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
    
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    root_dir = config.root_dir
    model_name = config.model_name
    
    new_model_name = '/'.join([model_name, f'a0'])
    modify_config(
        config,
        model_name=new_model_name,
    )
    if config.routine.compute_return_at_once:
        config.buffer.sample_keys += ['advantage', 'v_target']
    builder = ElementsBuilder(
        config, 
        env_stats, 
        to_save_code=False, 
        max_steps=config.routine.MAX_STEPS
    )
    elements = builder.build_agent_from_scratch()
    model = agent = elements.agent
    buffer = elements.buffer
    # if seed == 0:
    #     save_code(ModelPath(root_dir, model_name))
    
    routine_config = config.routine.copy()
    train(
        agent,
        model,
        runner,
        buffer,
        routine_config,
    )
    do_logging('Training completed')