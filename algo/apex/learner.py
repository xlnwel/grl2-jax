import threading
import tensorflow as tf
import ray

from core.ensemble import Ensemble
from core.tf_config import configure_gpu, configure_threads
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_gym_env
from replay.data_pipline import RayDataset


def create_learner(BaseAgent, name, model_fn, replay, config, model_config, env_config, replay_config):
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


    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    config['model_name'] = 'learner'
    # learner only define a env to get necessary env info, 
    # it does not actually interact with env
    env_config['n_workers'] = env_config['n_envs'] = 1

    if env_config.get('is_deepmind_env'):
        RayLearner = ray.remote(num_cpus=1, num_gpus=.5)(Learner)
    else:
        if tf.config.list_physical_devices('GPU'):
            RayLearner = ray.remote(num_cpus=1, num_gpus=.1)(Learner)
        else:
            RayLearner = ray.remote(num_cpus=1)(Learner)
    learner = RayLearner.remote(name, model_fn, replay, config, 
                            model_config, env_config)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=replay_config
    )))

    return learner
