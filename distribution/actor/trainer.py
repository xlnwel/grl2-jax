import time
import threading
import psutil

from core.tf_config import *
from core.dataset import create_dataset
from distribution.actor.base import RayBase
from utility.utils import config_attr
from utility.ray_setup import config_actor
from utility import pkg
from replay.func import create_replay
from env.func import create_env
from utility.typing import AttrDict


class Trainer(RayBase):
    def __init__(self, env_stats) -> None:
        self.trainers = {}
        self._env_stats = env_stats

    """ Only implements minimal functionality for learners """
    def start_training(self, aid):
        self._train_thread = threading.Thread(
            target=self._train, aid=aid, daemon=True)
        self._train_thread.start()

    def add_trainer(self, tid, trainer):
        self.trainers[tid] = trainer
    
    def set_weights(self, tid, weights):
        self.trainers[tid].set_weights(weights)

    def set_model_weights(self, tid, weights):
        self.trainers[tid].model.set_weights(weights)

    def get_weights(self, tid):
        return self.trainers[tid].get_weights()
    
    def get_model_weights(self, tid):
        return self.trainers[tid].model.get_weights()

    def _train(self, aid):
        if not hasattr(self, 'buffer'):
            raise RuntimeError(f'No buffer has been associate to trainer')
        self.dataset = self._create_dataset(self.buffer, )
        # waits for enough data to train
        while hasattr(self.dataset, 'good_to_learn') \
                and not self.dataset.good_to_learn():
            time.sleep(1)
        print(f'{self.name} starts learning...')

        while True:
            self.train_record()

    def _create_dataset(self, buffer, model, config, replay_config):
        am = pkg.import_module('agent', config=config, place=-1)
        data_format = am.get_data_format(
            env_stats=self._env_stats, 
            replay_config=replay_config, 
            agent_config=config, 
            model=model)
        dataset = create_dataset(
            buffer, self._env_stats, 
            data_format=data_format, 
            use_ray=getattr(self, '_use_central_buffer', True))
        
        return dataset

    def get_weights(self, name=None):
        return self.model.get_weights(name=name)

    def get_train_step_weights(self, name=None):
        return self.train_step, self.model.get_weights(name=name)

    def get_stats(self):
        """ retrieve training stats for the monitor to record """
        return self.train_step, super().get_stats()

    def set_handler(self, **kwargs):
        config_attr(self, kwargs)
    
    def get_weights(self, weights):
        


def get_learner_class(AgentBase):
    LearnerBase = get_learner_base_class(AgentBase)
    class Learner(LearnerBase):
        def __init__(self,
                    model_fn,
                    replay,
                    config, 
                    model_config,
                    env_config,
                    replay_config):
            name = 'Learner'
            psutil.Process().nice(config.get('default_nice', 0))

            config_actor(name, config)

            # avoids additional workers created by RayEnvVec
            env_config['n_workers'] = 1
            env_config['n_envs'] = 1
            env = create_env(env_config)

            model = model_fn(config=model_config, env=env)

            dataset = self._create_dataset(
                replay, model, env, config, replay_config) 
            
            super().__init__(
                name=name,
                config=config, 
                models=model,
                dataset=dataset,
                env=env,
            )

            env.close()

        def merge(self, data):
            assert hasattr(self, 'replay'), f'There is no replay in {self.name}.\nDo you use a central replay?'
            self.replay.merge(data)
        
        def good_to_learn(self):
            assert hasattr(self, 'replay'), f'There is no replay in {self.name}.\nDo you use a central replay?'
            return self.replay.good_to_learn()

    return Learner
