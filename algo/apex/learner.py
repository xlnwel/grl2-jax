import threading
import tensorflow as tf
import ray

from core.ensemble import Ensemble
from core.tf_config import configure_gpu, configure_threads
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_gym_env
from replay.data_pipline import RayDataset


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

            env = create_gym_env(env_config)
            data_format = dict(
                state=(env.state_dtype, (None, *env.state_shape)),
                action=(env.action_dtype, (None, *env.action_shape)),
                reward=(tf.float32, (None, )), 
                next_state=(env.state_dtype, (None, *env.state_shape)),
                done=(tf.float32, (None, )),
                steps=(tf.float32, (None, )),
            )
            if config['algorithm'].endswith('il'):
                data_format.update(dict(
                    mu=(tf.float32, (None, *env.action_shape)),
                    std=(tf.float32, (None, *env.action_shape)),
                    logpi=(tf.float32, (None, 1)),
                    kl_flag=(tf.float32, (None, 1)),
                ))
            dataset = RayDataset(replay, data_format)
            self.model = Ensemble(model_fn, model_config, env.state_shape, env.action_dim, env.is_action_discrete)
            
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
            self.writer.set_as_default()
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
