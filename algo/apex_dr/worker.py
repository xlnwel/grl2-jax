from collections import deque
import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.ensemble import Ensemble
from utility.display import pwc
from utility.timer import Timer, TBTimer
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

        self.threshold = -float('inf')

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
        while step < 1e7:
            self.set_summary_step(step)
            with Timer(f'{self.name} pull weights', self.TIME_PERIOD):
                weights = self.pull_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_PERIOD):
                step, scores, epslens = self.eval_model(weights, step, replay)

            with Timer(f'{self.name} send data', self.TIME_PERIOD):
                self._send_data(replay, scores, epslens)

            self._periodic_logging(step)
        
    def _send_data(self, replay, scores=None, epslens=None):
        self.threshold = max(np.max(scores) - self.SLACK, self.threshold)
        env_mask = scores > self.threshold

        """ sends data to replay """
        mask, data = self.buffer.sample()
        data_tesnors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        data['priority'] = self.compute_priorities(**data_tesnors).numpy()

        # squeeze since many terms in data is of shape [None, 1]
        for k, v in data.items():
            data[k] = np.squeeze(v)

        good_mask = np.zeros_like(mask, dtype=np.bool)
        good_mask[env_mask] = 1
        good_mask = good_mask[mask]
        regular_mask = (1 - good_mask).astype(np.bool)

        good_data = {}
        regular_data = {}
        for k, v in data.items():
            good_data[k] = v[good_mask]
            regular_data[k] = v[regular_mask]
        self.store(
            good_frac=np.mean(env_mask), 
            threshold=self.threshold,
            good_scores=np.mean(scores[env_mask]) if np.any(env_mask) else 0,
            good_epslens=np.mean(epslens[env_mask]) if np.any(env_mask) else 0,
        )
        if np.any(env_mask):
            self.store(
                good_scores=np.mean(scores[env_mask]) if np.any(env_mask) else 0,
                good_epslens=np.mean(epslens[env_mask]) if np.any(env_mask) else 0,
            )
        if not np.all(env_mask):
            self.store(
                regular_scores=np.mean(scores[(1-env_mask).astype(np.bool)]),
                regular_epslens=np.mean(epslens[(1-env_mask).astype(np.bool)]),
            )
        
        
        if np.any(env_mask):
            replay.merge.remote(good_data, good_data['state'].shape[0], 'good_replay')
        if not np.all(env_mask):
            replay.merge.remote(regular_data, regular_data['state'].shape[0], 'regular_replay')

        self.buffer.reset()


def create_worker(name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    model_config['actor']['gamma'] = config['gamma']

    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_fn = create_local_buffer

    env_config['seed'] = worker_id
    
    config['model_name'] = f'worker_{worker_id}'
    config['TIME_PERIOD'] = 1000
    config['LOG_STEPS'] = 10000
    config['SLACK'] = 10

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