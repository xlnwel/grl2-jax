import random
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
from algo.asap.utils import *


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
        buffer = buffer_fn(buffer_config)

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
        log_time = self.LOG_INTERVAL
        while step < self.MAX_STEPS:
            with TBTimer(f'{self.name} pull weights', self.TIME_INTERVAL, to_log=self.timer):
                mode, score, tag, weights, eval_times = self.pull_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_INTERVAL, to_log=self.timer):
                step, scores, epslens = self.eval_model(
                    weights, step, replay, tag=tag, store_exp=mode != Mode.REEVALUATION)
            eval_times = eval_times + self.n_envs

            self._log_episodic_info(tag, scores, epslens)

            # if mode != Mode.REEVALUATION:
            with TBTimer(f'{self.name} send data', self.TIME_INTERVAL, to_log=self.timer):
                self._send_data(replay, tag)

            score += self.n_envs / eval_times * (np.mean(scores) - score)

            learner.store_weights.remote(self.id, score, eval_times)
            
            if self.env.name == 'BipedalWalkerHardcore-v2' and eval_times > 100 and score > 300:
                self.save(print_terminal_info=False)
            elif step > log_time and tag == Tag.EVOLVED:
                self.save(print_terminal_info=False)
                log_time += self.LOG_INTERVAL

    def _log_episodic_info(self, tag, scores, epslens):
        if scores is not None:
            if tag == 'Learned':
                self.store(
                    score=scores,
                    epslen=epslens,
                )
            else:
                self.store(
                    evolved_score=scores,
                    evolved_epslen=epslens,
                )

    def _log_condition(self):
        return self.logger.get_count('score') > 0 and self.logger.get_count('evolved_score') > 0

    def _logging(self, step):
        # record stats
        self.store(**self.get_value('score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('epslen', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('evolved_score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('evolved_epslen', mean=True, std=True, min=True, max=True))
        self.log(step, print_terminal_info=False)
    

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

    if env_config.get('is_deepmind_env'):
        RayWorker = ray.remote(num_cpus=2)(Worker)
    else:
        RayWorker = ray.remote(num_cpus=1)(Worker)
    worker = RayWorker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    ray.get(worker.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=buffer_config
    )))

    return worker
