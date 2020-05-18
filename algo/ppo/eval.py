import numpy as np
import tensorflow as tf
import ray

from utility.display import pwc
from utility.graph import save_video
from utility.run import evaluate
from core.tf_config import *
from env.gym_env import create_env
from run import pkg


def main(env_config, model_config, agent_config, n, record=False, size=(128, 128)):
    silence_tf_logs()
    configure_gpu()

    algo = agent_config['algorithm']
    create_model, Agent = pkg.import_agent(agent_config)

    env = create_env(env_config, force_envvec=True)

    models = create_model(
        model_config, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
    )

    agent = Agent(name=algo, config=agent_config, models=models, env=env)

    scores, epslens, video = evaluate(env, agent, n, record=record, size=size)
    if record:
        save_video(agent._model_name, video)
    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')