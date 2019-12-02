from collections import deque
import numpy as np
import tensorflow as tf
import ray

from core import log
from core import tf_config
from utility.display import pwc
from env.gym_env import create_gym_env
from algo.apex_sac.buffer import create_local_buffer
from algo.apex_sac.per_worker import PERWorker


@ray.remote(num_cpus=1)
class Worker(PERWorker):
    """ Interface """
    def __init__(self, 
                name,
                worker_id, 
                model_fn,
                buffer_fn,
                config,
                model_config, 
                env_config, 
                buffer_config):
        tf_config.configure_threads(1, 1)
        tf_config.configure_gpu()

        env = create_gym_env(env_config)
        
        models = model_fn(model_config, env.state_shape, env.action_dim, env.is_action_discrete)
        
        buffer_config['epslen'] = env.max_episode_steps
        buffer = buffer_fn(
            buffer_config, env.state_shape, 
            env.state_dtype, env.action_dim, 
            env.action_dtype, config['gamma'])

        super().__init__(
            name=name,
            worker_id=worker_id,
            models=models,
            env=env,
            buffer=buffer,
            actor=models['actor'],
            value=models['q1'],
            target_value=models['target_q1'],
            config=config)

    def run(self, learner):
        episode_i = 0
        step = 0
        while True:
            weights = self.pull_weights(learner)
            episode_i, step, _ = self.eval_model(weights, episode_i, step)
            self.send_data(learner)


def create_worker(name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    config['model_name'] = f'worker_{worker_id}'
    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_config['type'] = 'env' if buffer_config['n_envs'] == 1 else 'envvec'
    buffer_fn = create_local_buffer
    env_config['n_workers'] = 1
    env_config['seed'] = 100 * worker_id

    return Worker.remote(
        name,
        worker_id, 
        model_fn,
        buffer_fn,
        config,
        model_config, 
        env_config, 
        buffer_config)
