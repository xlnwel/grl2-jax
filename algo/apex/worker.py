from collections import deque
import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.ensemble import Ensemble
from utility.display import pwc
from utility.timer import Timer
from env.gym_env import create_gym_env
from algo.apex.buffer import create_local_buffer
from algo.apex.base_worker import BaseWorker


@ray.remote(num_cpus=1)
class Worker(BaseWorker):
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
        
        models = Ensemble(model_fn, model_config, env.state_shape, env.action_dim, env.is_action_discrete)
        
        buffer_config['seqlen'] = env.max_episode_steps
        buffer = buffer_fn(
            buffer_config, env.state_shape, 
            env.state_dtype, env.action_shape, 
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

    def run(self, learner, replay):
        episode = 0
        step = 0
        while True:
            with Timer(f'{self.name} pull weights', self.TIME_PERIOD):
                weights = self.pull_weights(learner)
            episode, step, _ = self.eval_model(weights, episode, step, replay)

            self._periodic_logging(episode, step)

    def _periodic_logging(self, episode, step):
        if episode % self.LOG_PERIOD == 0:
            self.log(step=step, print_terminal_info=False)

def create_worker(name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_config['type'] = 'env' if buffer_config['n_envs'] == 1 else 'envvec'
    buffer_fn = create_local_buffer

    env_config['efficient_envvec'] = True
    env_config['seed'] = 100 * worker_id
    
    config['model_name'] = f'worker_{worker_id}'
    config['LOG_PERIOD'] = 20
    config['TIME_PERIOD'] = 1000

    name = f'{name}_{worker_id}'
    worker = Worker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    ray.get(worker.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=buffer_config
    )))

    return worker