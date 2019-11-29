from collections import deque
import numpy as np
import tensorflow as tf
import ray

from core import log
from core.tf_config import configure_gpu, configure_threads
from utility.display import pwc
from algo.apex_sac.buffer import create_local_buffer
from algo.apex_sac.base_worker import BaseWorker


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
                buffer_config, 
                to_log):
        super().__init__(
            name=name,
            worker_id=worker_id,
            model_fn=model_fn,
            buffer_fn=buffer_fn, 
            config=config,
            model_config= model_config, 
            env_config=env_config, 
            buffer_config= buffer_config, 
            to_log=to_log)

    def run(self, learner):
        episode_i = 0
        step = 0
        while True:
            weights = self._pull_weights(learner)
            episode_i, step, score_mean = self.eval_model(weights, episode_i, step)
            self.send_data(learner)


def create_worker(name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config, to_log):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    config['model_name'] += f'_worker_{worker_id}'
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
        buffer_config, 
        to_log)
