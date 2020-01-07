import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.ensemble import Ensemble
from utility.display import pwc
from utility.timer import TBTimer
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
            config=config)

    def run(self, learner, replay):
        step = 0
        i = 0
        while step < self.MAX_STEPS:
            i += 1
            self.set_summary_step(step)
            
            with TBTimer(f'{self.name} pull weights', self.TIME_INTERVAL, to_log=self.timer):
                weights = self.pull_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_INTERVAL, to_log=self.timer):
                step, _, _ = self.eval_model(weights, step, replay)

            with TBTimer(f'{self.name} send data', self.TIME_INTERVAL, to_log=self.timer):
                self._send_data(replay)

            self._periodic_logging(step, i)


def create_worker(name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_fn = create_local_buffer

    env_config['seed'] += worker_id * 100
    
    config['model_name'] = f'worker_{worker_id}'
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
