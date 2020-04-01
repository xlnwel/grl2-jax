import numpy as np
import tensorflow as tf
import ray

from utility.display import pwc
from utility.utils import set_global_seed
from core.tf_config import configure_gpu, silence_tf_logs
from utility.signal import sigint_shutdown_ray
from env.gym_env import create_env


def import_model_fn(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.nn import create_model
    elif algorithm == 'ppo1':
        from algo.ppo.nn import create_model
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

def evaluate(env, agent, n=1):
    pwc('Evaluation starts', color='cyan')
    scores = []
    epslens = []
    for i in range(0, n, env.n_envs):
        agent.reset_states()
        obs = env.reset()
        for _ in range(env.max_episode_steps):
            action = agent(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            if np.all(done):
                break
            
        scores.append(env.score())
        epslens.append(env.epslen())

    return scores, epslens

def main(env_config, model_config, agent_config, render=False):
    algo = agent_config['algorithm']
    create_model = import_model_fn(algo)
    Agent = import_agent(algo)

    silence_tf_logs()
    set_global_seed()
    configure_gpu()

    env = create_env(env_config, force_envvec=True)

    models = create_model(
        model_config, 
        obs_shape=env.obs_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
        n_envs=env.n_envs
    )

    agent = Agent(name=algo, config=agent_config, models=models, env=env)
    
    agent.restore()

    scores, epslens = evaluate(env, agent)
    pwc(f'After running 100 episodes',
        f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')