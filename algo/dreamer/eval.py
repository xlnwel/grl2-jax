import numpy as np

from core.tf_config import *
from env.gym_env import create_env
from algo.sac.run import evaluate
from algo.dreamer.env import make_env
from algo.dreamer.nn import create_model
from algo.dreamer.agent import Agent
from utility.graph import save_video


def main(env_config, model_config, agent_config, n, render=False):
    silence_tf_logs()
    configure_gpu()
    
    env = create_env(env_config, make_env, force_envvec=True)

    models = create_model(
        model_config,
        obs_shape=env.obs_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
    )

    agent = Agent(
        name=agent_config['algorithm'], 
        config=agent_config, 
        models=models, 
        dataset=None, 
        env=env)
    
    agent.restore()

    scores, epslens = evaluate(env, agent, n, render=render)
    save_video('dreamer', env.prev_episode['obs'][None])
    
    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')
