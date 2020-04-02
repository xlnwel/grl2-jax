import threading
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy
import ray

from core.module import Ensemble
from core.tf_config import *
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_env
from replay.data_pipline import process_with_env, DataFormat, RayDataset


def get_learner_class(BaseAgent):
    class Learner(BaseAgent):
        """ Interface """
        def __init__(self,
                    name, 
                    model_fn,
                    replay,
                    config, 
                    model_config,
                    env_config):
            # tf.debugging.set_log_device_placement(True)
            silence_tf_logs()
            configure_threads(1, 1)
            configure_gpu()
            configure_precision(getattr(self, '_precision', 32))
            self._dtype = global_policy().compute_dtype

            env = create_env(env_config)
            data_format = dict(
                obs=DataFormat((None, *env.obs_shape), self._dtype),
                action=DataFormat((None, *env.action_shape), self._dtype),
                reward=DataFormat((None, ), self._dtype), 
                next_obs=DataFormat((None, *env.obs_shape), self._dtype),
                done=DataFormat((None, ), self._dtype),
            )
            if ray.get(replay.buffer_type.remote()).endswith('proportional'):
                data_format['IS_ratio'] = DataFormat((None, ), self._dtype)
                data_format['saved_idxes'] = DataFormat((None, ), tf.int32)
            if config['n_steps'] > 1:
                data_format['steps'] = DataFormat((None, ), self._dtype)
            if config['algorithm'].endswith('il'):
                data_format.update(dict(
                    mu=DataFormat((None, *env.action_shape), self._dtype),
                    std=DataFormat((None, *env.action_shape), self._dtype),
                    kl_flag=DataFormat((None, ), self._dtype),
                ))
            print(data_format)
            process = functools.partial(process_with_env, env=env)
            dataset = RayDataset(replay, data_format, process)

            self.models = Ensemble(
                model_fn=model_fn, 
                model_config=model_config, 
                action_dim=env.action_dim, 
                is_action_discrete=env.is_action_discrete)

            super().__init__(
                name=name, 
                config=config, 
                models=self.models,
                dataset=dataset,
                env=env,
            )
            
        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            pwc(f'{self.name} starts learning...', color='blue')
            step = 0
            self._writer.set_as_default()
            while True:
                step += 1
                self.learn_log(step)
                if step % 1000 == 0:
                    self.log(step, print_terminal_info=False)
                if step % 100000 == 0:
                    self.save(print_terminal_info=False)

        def get_weights(self, worker_id, name=None):
            return self.models.get_weights(name=name)

    return Learner
