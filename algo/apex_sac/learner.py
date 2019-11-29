import time
import threading
import numpy as np
import tensorflow as tf
import ray

from core.tf_config import configure_gpu, configure_threads
from utility.display import pwc
from utility.timer import Timer
from env.gym_env import create_gym_env
from replay.func import create_replay
from algo.sac.data_pipline import Dataset


def create_learner(BaseAgent, name, model_fn, config, model_config, env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()
    
    config['model_name'] += '_learner'
    env_config['n_workers'] = env_config['n_envs'] = 1
    
    @ray.remote(num_gpus=0.3, num_cpus=2)
    class Learner(BaseAgent):
        """ Interface """
        def __init__(self,
                    name, 
                    model_fn,
                    config, 
                    model_config,
                    env_config,
                    buffer_config):
            # tf.debugging.set_log_device_placement(True)
            configure_threads(2, 2)
            configure_gpu()

            env = create_gym_env(env_config)
            self.buffer = create_replay(
                buffer_config, env.state_shape, 
                env.state_dtype, env.action_dim, 
                env.action_dtype, config['gamma'], 
                has_next_state=True)
            dataset = Dataset(self.buffer, env.state_shape, env.action_dim)
            self.model = model_fn(model_config, env.state_shape, env.action_dim, env.is_action_discrete)
            
            super().__init__(
                name=name, 
                config=config, 
                models=self.model,
                dataset=dataset,
                state_shape=env.state_shape,
                state_dtype=env.state_dtype,
                action_dim=env.action_dim,
                action_dtype=env.action_dtype,
            )
            
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def get_weights(self):
            return self.model.get_weights()

        def merge_buffer(self, local_buffer, length):
            self.buffer.merge(local_buffer, length)

        def _learning(self):
            while not self.buffer.good_to_learn:
                time.sleep(1)
            pwc(f'{self.name} starts learning...', color='blue')

            step = 0
            self.writer.set_as_default()
            while True:
                step += 1
                self.train_log()
                if step % 1000 == 0:
                    self.log_summary(self.logger.get_stats(), step)
                    self.save(steps=step)


    return Learner.remote(name, model_fn, config, model_config, env_config, buffer_config)
