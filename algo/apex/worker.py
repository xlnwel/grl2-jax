import numpy as np
import tensorflow as tf
import ray

from core.tf_config import *
from core.module import Ensemble
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_env
from algo.apex.buffer import create_local_buffer
from algo.apex.base_worker import BaseWorker


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
        silence_tf_logs()
        configure_threads(1, 1)
        configure_gpu()

        env = create_env(env_config)
        
        buffer_config['seqlen'] = env.max_episode_steps
        buffer = buffer_fn(buffer_config)

        models = Ensemble(
            model_fn=model_fn, 
            model_config=model_config, 
            action_dim=env.action_dim, 
            is_action_discrete=env.is_action_discrete)

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
            weights = self.pull_weights(learner)

            step, scores, epslens = self.eval_model(weights, step)

            self._log_episodic_info(scores, epslens)

            self._send_data(replay)

            score = np.mean(scores)
            
            if step > log_time:
                self.save(print_terminal_info=False)
                log_time += self.LOG_INTERVAL

    def _log_episodic_info(self, scores, epslens):
        if scores is not None:
            self.store(
                score=scores,
                epslen=epslens,
            )

def get_worker_class():
    return Worker