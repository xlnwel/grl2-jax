from collections import deque
import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.ensemble import Ensemble
from utility.display import pwc
from utility.timer import TBTimer, TBTimer
from env.gym_env import create_gym_env
from algo.apex.buffer import create_local_buffer
from algo.apex.base_worker import BaseWorker


LOG_STEPS = 10000

@ray.remote#(num_cpus=1)
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
        step = 0
        while step < self.MAX_STEPS:
            self.set_summary_step(step)
            with TBTimer(f'{self.name} pull weights', self.TIME_PERIOD, to_log=self.timer):
                weights = self.pull_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_PERIOD, to_log=self.timer):
                step, _, _ = self.eval_model(weights, step, replay)

            with TBTimer(f'{self.name} send data', self.TIME_PERIOD, to_log=self.timer):
                self._send_data(replay)

            self._periodic_logging(step)

    def _send_data(self, replay):
        """ sends data to replay """
        mask, data = self.buffer.sample()
        data_tesnors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        if not self.replay_type.endswith('uniform'):
            data['priority'] = self.compute_priorities(**data_tesnors).numpy()
        
        # squeeze since many terms in data is of shape [None, 1]
        for k, v in data.items():
            data[k] = np.squeeze(v)

        replay.merge.remote(data, data['state'].shape[0])

        self.buffer.reset()


def create_worker(name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_fn = create_local_buffer

    env_config['seed'] = 100 * worker_id
    
    config['model_name'] = f'worker_{worker_id}'
    config['TIME_PERIOD'] = 1000
    config['LOG_STEPS'] = 10000
    config['MAX_STEPS'] = int(1e8)
    config['replay_type'] = buffer_config['type']

    worker = Worker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    ray.get(worker.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=buffer_config
    )))

    return worker