import threading
import tensorflow as tf
import ray

from core.module import Ensemble
from core.tf_config import configure_gpu, configure_threads, configure_precision
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_env
from replay.data_pipline import DataFormat, RayDataset


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
            configure_threads(1, 1)
            configure_gpu()
            configure_precision(getattr(self, 'precision', 16))

            env = create_env(env_config)
            data_format = dict(
                obs=DataFormat((None, *env.obs_shape), env.obs_dtype),
                action=DataFormat((None, *env.action_shape), env.action_dtype),
                reward=DataFormat((None, ), tf.float32), 
                next_obs=DataFormat((None, *env.obs_shape), env.obs_dtype),
                done=DataFormat((None, ), tf.float32),
            )
            if replay.n_steps > 1:
                data_format['steps'] = DataFormat((None, ), tf.float32)
            if config['algorithm'].endswith('il'):
                data_format.update(dict(
                    mu=(tf.float32, (None, *env.action_shape)),
                    std=(tf.float32, (None, *env.action_shape)),
                    kl_flag=(tf.float32, (None, )),
                ))
            dataset = RayDataset(replay, data_format)
            self.model = Ensemble(model_fn, model_config, env.obs_shape, env.action_dim, env.is_action_discrete)
            
            super().__init__(
                name=name, 
                config=config, 
                models=self.model,
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
                with TBTimer(f'{self.name} train', 10000, to_log=self.timer):
                    self.learn_log(step)
                if step % 1000 == 0:
                    self.log(step, print_terminal_info=False)
                if step % 100000 == 0:
                    self.save(print_terminal_info=False)

        def get_weights(self, worker_id, name=None):
            return self.model.get_weights(name=name)

    return Learner
