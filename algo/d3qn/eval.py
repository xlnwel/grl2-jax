import numpy as np
import tensorflow as tf
import ray

from core.tf_config import *
from utility.display import pwc
from utility.graph import save_video
from utility.ray_setup import sigint_shutdown_ray
from utility.run import evaluate
from env.gym_env import create_env
from algo.d3qn.nn import create_model
from algo.d3qn.agent import Agent


def main(env_config, model_config, agent_config, n, record=False):
    silence_tf_logs()
    configure_gpu()

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()
        
    if record:
        env_config['n_workers'] = env_config['n_envs'] = 1

    env = create_env(env_config)
    
    # construct models
    models = create_model(model_config, env.action_dim)

    # construct agent
    agent = Agent(name='q', 
                config=agent_config, 
                models=models, 
                dataset=None, 
                env=env)


    results = evaluate(env, agent, n, record=record)
    if record:
        scores, epslens, video = results
        save_video('q', video)
    else:
        scores, epslens, = results
    
    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')

    ray.shutdown()