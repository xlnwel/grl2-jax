import numpy as np
import tensorflow as tf
import ray

from utility.display import pwc
from utility.run import evaluate
from core.tf_config import *
from env.gym_env import create_env


def import_model_fn(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.nn import create_model
    elif algorithm == 'ppo1':
        from algo.ppo1.nn import create_model
    elif algorithm == 'ppo2':
        from algo.ppo2.nn import create_model
    elif algorithm == 'ppo3':
        from algo.ppo3.nn import create_model
    else:
        raise NotImplementedError(algorithm)
    return create_model

def import_agent(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.agent import Agent
    elif algorithm == 'ppo1':
        from algo.ppo1.agent import Agent
    elif algorithm == 'ppo2':
        from algo.ppo2.agent import Agent
    elif algorithm == 'ppo3':
        from algo.ppo3.agent import Agent
    else:
        raise NotImplementedError(algorithm)
    return Agent

def main(env_config, model_config, agent_config, n, record=False):
    silence_tf_logs()
    configure_gpu()

    algo = agent_config['algorithm']
    create_model = import_model_fn(algo)
    Agent = import_agent(algo)

    if record:
        env_config['n_workers'] = env_config['n_envs'] = 1
    env = create_env(env_config, force_envvec=True)

    models = create_model(
        model_config, 
        obs_shape=env.obs_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
        n_envs=env.n_envs
    )

    agent = Agent(name=algo, config=agent_config, models=models, env=env)

    scores, epslens, video = evaluate(env, agent, n, record=record)
    pwc(f'After running 100 episodes',
        f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')