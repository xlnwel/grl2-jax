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
        buffer_keys = ['state', 'action', 'reward', 'done', 'steps']
        buffer = buffer_fn(buffer_config, *buffer_keys)

        super().__init__(
            name=name,
            worker_id=worker_id,
            models=models,
            env=env,
            buffer=buffer,
            actor=models['actor'],
            value=models['q1'],
            config=config)

        self.raw_bookkeeping = BookKeeping('raw')
        self.best_score = -float('inf')
        # self.reevaluation_bookkeeping = BookKeeping('reeval')

    def run(self, learner, replay):
        step = 0
        while step < self.MAX_STEPS:
            with TBTimer(f'{self.name} pull weights', self.TIME_INTERVAL, to_log=self.timer):
                threshold, mode, score, tag, weights, eval_times = self.pull_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_INTERVAL, to_log=self.timer):
                step, scores, epslens = self.eval_model(weights, step, replay, 
                                                        evaluation=mode == Mode.REEVALUATION, tag=tag)
            eval_times = eval_times + self.n_envs

            with TBTimer(f'{self.name} send data', self.TIME_INTERVAL, to_log=self.timer):
                self._send_data(replay)

            score += self.n_envs / eval_times * (np.mean(scores) - score)
            self.best_score = max(self.best_score, score)

            status = self._make_decision(threshold, score, tag, eval_times)

            if status == Status.ACCEPTED:
                learner.store_weights.remote(score, tag, weights, eval_times)
            
            if self.env.name == 'BipedalWalkerHardcore-v2' and eval_times > 100 and score > 300:
                self.save()
            elif score == self.best_score:
                self.save()

    def _make_decision(self, threshold, score, tag, eval_times):
        if score > threshold:
            status = Status.ACCEPTED
        else:
            status = Status.REJECTED
        self.raw_bookkeeping.add(tag, status)

        pwc(f'{self.name}_{self.id}: {tag} model has been evaluated({eval_times}).', 
            f'Score: {score:.3g}',
            f'Decision: {status}', color='green')

        return status

    def _log_condition(self):
        return self.logger.get_count('score') > 0 and self.logger.get_count('evolved_score') > 0

    def _logging(self, step):
        # record stats
        self.store(**self.raw_bookkeeping.stats())
        # self.store(**self.reevaluation_bookkeeping.stats())
        self.raw_bookkeeping.reset()
        # self.reevaluation_bookkeeping.reset()
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

    worker = Worker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    ray.get(worker.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=buffer_config
    )))

    return worker
