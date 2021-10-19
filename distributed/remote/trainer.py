import time
import threading
import psutil

from core.tf_config import *
from core.dataset import create_dataset
from utility import pkg
from utility.utils import config_attr
from utility.ray_setup import config_actor
from env.func import create_env
from distributed.remote.base import RayBase


class Trainer(RayBase):
    def __init__(self, env_stats, aid=None):
        self._env_stats = env_stats
        self._aid = aid

        self.model_constructors = {}
        self.loss_constructors = {}
        self.trainer_constructors = {}

        self.trainers = {}
        self.configs = {}
        # we defer all constructions to the run time

    def start_training(self, aid):
        self._train_thread = threading.Thread(
            target=self._train, aid=aid, daemon=True)
        self._train_thread.start()

    def add_trainer(self, sid, trainer):
        self.trainers[sid] = trainer
    
    def set_weights(self, sid, weights):
        self.trainers[sid].set_weights(weights)

    def set_model_weights(self, sid, weights):
        self.trainers[sid].model.set_weights(weights)

    def get_weights(self, sid):
        return self.trainers[sid].get_weights()
    
    def get_model_weights(self, sid):
        return self.trainers[sid].model.get_weights()

    def construct_trainer_from_config(self, config, sid=None):
        algo = config.algorithm
        self._setup_constructors(algo)
        trainer = self._construct_trainer(algo, config, self._env_stats)
        
        identifier = f'{self._aid}_{algo}' \
            if self._aid is not None else algo
        if sid is not None:
            identifier = f'{identifier}_{sid}'
        self.trainers[identifier] = trainer
        self.configs[identifier] = config

    def _setup_constructors(self, algo):
        if algo in self.trainer_constructors:
            return
        self.model_constructors[algo] = pkg.import_module(
            name='elements.model', algo=algo, place=-1).create_model
        self.loss_constructors[algo] = pkg.import_module(
            name='elements.loss', algo=algo, place=-1).create_loss
        self.trainer_constructors[algo] = pkg.import_module(
            name='elements.trainer', algo=algo, place=-1).create_trainer

    def _construct_trainer(self, algo, config, env_stats):
        model = self.model_constructors[algo](config.model, env_stats)
        loss = self.loss_constructors[algo](config.loss, model)
        trainer = self.trainer_constructors[algo](
            config.trainer, model, loss, env_stats)

        return trainer

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
        am = pkg.import_module('elements.utils', config=config, place=-1)
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
        
