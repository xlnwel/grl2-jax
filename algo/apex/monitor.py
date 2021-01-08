import time
import threading
import numpy as np
import tensorflow as tf
import ray

from core.decorator import record
from core.base import AgentImpl
from utility.graph import video_summary


class Monitor(AgentImpl):
    @record
    def __init__(self, config):
        self._ready = np.zeros(config['n_workers'])
        self._locker = threading.Lock()

        self.time = time.time()
        self.env_step = 0
        self.last_env_step = 0
        self.last_train_step = 0
        self.MAX_STEPS = int(float(config['MAX_STEPS']))
    
    def record_episode_info(self, worker_id=None, **stats):
        video = stats.pop('video', None)
        if 'epslen' in stats:
            self.env_step += np.sum(stats['epslen'])
        if worker_id is not None:
            stats = {f'{k}_{worker_id}': v for k, v in stats.items()}
        with self._locker:
            self.store(**stats)
        if video is not None:
            video_summary(f'{self.name}/sim', video, step=self.env_step)

    def record_stats(self, learner):
        train_step, stats = ray.get(learner.get_stats.remote())
        if train_step == 0:
            return
        duration = time.time() - self.time
        self.store(
            train_step=train_step, 
            env_step=self.env_step, 
            fps=(self.env_step - self.last_env_step) / duration,
            tps=(train_step - self.last_train_step) / duration,
            **stats)
        self.log(self.env_step)
        self.last_train_step = train_step
        self.last_env_step = self.env_step
        learner.save.remote()
    
    def is_over(self):
        return self.env_step > self.MAX_STEPS
