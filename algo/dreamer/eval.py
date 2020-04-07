import numpy as np
import ray

from core.tf_config import *
from utility.ray_setup import sigint_shutdown_ray
from utility.graph import save_video
from env.gym_env import create_env
from algo.common.run import evaluate
from algo.dreamer.env import make_env
from algo.dreamer.nn import create_model
from algo.dreamer.agent import Agent


def main(env_config, model_config, agent_config, n, render=False):
    silence_tf_logs()
    configure_gpu()
    
    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init(num_cpus=12, num_gpus=1)
        sigint_shutdown_ray()

    if render:
        env_config['log_episode'] = True
        env_config['n_workers'] = env_config['n_envs'] = 1
    env = create_env(env_config, make_env, force_envvec=True)

    models = create_model(
        model_config,
        obs_shape=env.obs_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
    )

    agent_config['store_state'] = False
    agent = Agent(
        name=agent_config['algorithm'], 
        config=agent_config, 
        models=models, 
        dataset=None, 
        env=env)

    scores, epslens = evaluate(env, agent, n, render=render)
    if render:
        save_video('dreamer', env.prev_episode['obs'][None])
    
    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')
