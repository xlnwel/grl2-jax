import time
import threading
import ray
from ray.util.queue import Queue

from core.dataset import create_dataset
from core.mixin import IdentifierConstructor
from core.mixin.monitor import create_recorder
from core.tf_config import *
from core.utils import save_config
from core.remote.base import RayBase
from distributed.typing import WeightsTuple
from utility import pkg
from utility.display import pwc
from utility.ray_setup import config_actor
from utility.timer import Every
from utility.utils import AttrDict2dict, dict2AttrDict


class RemoteTrainer(RayBase):
    def __init__(self, config, env_stats, name=None):
        self.config = dict2AttrDict(config)
        self._env_stats = env_stats
        config_actor(name, self.config.coordinator.actor_config)
        self._name = name
        self._idc = IdentifierConstructor()

        self.model_constructor = None
        self.loss_constructor = None
        self.trainer_constructor = None
        self.strategy_constructor = None

        self.aid = None
        self.sid = None
        self.identifier = None
        self.model = None
        self.trainer = None
        self.strategy = None

        self.buffer = None
        self.dataset = None

        self.param_queues = []
        self.recorder = create_recorder(None, None)
        # we defer all constructions to the run time

    """ Construction """
    def construct_strategy_from_config(self, 
            config, aid=None, sid=None, weights=None, buffer=None):
        """ Construct a strategy from config """
        self.config = config = dict2AttrDict(config)
        self._setup_constructors(config)
        self._construct_trainer(config, self._env_stats)
        self._construct_dataset(config, buffer)
        self._construct_strategy(config)

        self.aid = aid
        self.sid = sid
        self.identifier = self._idc.get_identifier(aid, sid)
        if weights is not None:
            self.strategy.set_weights(weights, aid, sid)
        
        save_config(
            config.root_dir, config.model_name, 
            config, f'{aid}-{sid}')

    def _setup_constructors(self, config):
        algo = config.algorithm.split('-')[-1]
        self.model_constructor = pkg.import_module(
            name='elements.model', algo=algo, place=-1).create_model
        self.loss_constructor = pkg.import_module(
            name='elements.loss', algo=algo, place=-1).create_loss
        self.trainer_constructor = pkg.import_module(
            name='elements.trainer', algo=algo, place=-1).create_trainer
        self.strategy_constructor = pkg.import_module(
            name='elements.strategy', algo=algo, place=-1).create_strategy

    def _construct_trainer(self, config, env_stats):
        self.model = self.model_constructor(config.model, env_stats)
        loss = self.loss_constructor(config.loss, self.model)
        self.trainer = self.trainer_constructor(
            config.trainer, loss, env_stats)

    def _construct_dataset(self, config, buffer=None):
        algo = config.algorithm.split('-')[0]
        if buffer is None:
            buffer_constructor = pkg.import_module(
                'buffer', pkg=f'distributed.{algo}').create_central_buffer
            self.buffer = buffer_constructor(AttrDict2dict(config.buffer))
        else:
            self.buffer = buffer
        am = pkg.import_module('elements.utils', config=config, place=-1)
        data_format = am.get_data_format(
            config=config.trainer,
            env_stats=self._env_stats, 
            model=self.model)
        self.dataset = create_dataset(
            self.buffer, 
            env_stats=self._env_stats, 
            data_format=data_format, 
            central_buffer=True)

    def _construct_strategy(self, config):
        self.strategy = self.strategy_constructor(
            self.identifier, config.strategy, 
            self.trainer, dataset=self.dataset)

    """ Actor """
    def register_actor(self, actor):
        pq = Queue()
        actor.register_buffer.remote(self.aid, self.buffer)
        actor.register_param_queue.remote(self.aid, pq)
        self.param_queues.append(pq)

    """ Training """
    def start_training(self):
        self._train_thread = threading.Thread(
            target=self._train, daemon=True)
        self._train_thread.start()

    def _train(self):
        while self.strategy is None:
            time.sleep(1)

        if not hasattr(self, 'buffer'):
            raise RuntimeError(f'No buffer has been associate to trainer')
        pwc(f'{self._name} starts learning...', color='cyan')

        to_record = Every(
            self.config.monitor.LOG_PERIOD, 
            self.config.monitor.LOG_PERIOD)
        start_train_step = self.strategy.get_train_step()
        start_train_time = time.time()
        while True:
            stats = self.strategy.train_record()
            self.recorder.store(**stats)
            train_step = self.strategy.get_train_step()
            self.push_weights(train_step)
            if to_record(train_step):
                stats = self.recorder.get_stats()
                train_time = time.time()
                stats.update({
                    'train_step': train_step,
                    'tps': (train_step - start_train_step) / (train_time - start_train_time),
                })
                start_train_time = train_time
                start_train_step = train_step
                self.monitor.store_stats.remote(**stats)

    """ Get & Set """
    def push_weights(self, train_step=None):
        aux = self.buffer.get_aux_stats.remote()
        identifier = self._idc.get_identifier(aid=self.aid, sid=self.sid)
        weights = self.get_weights(identifier)
        weights.weights[f'{identifier}_aux'] = ray.get(aux)
        if train_step is None:
            train_step = self.strategy.get_train_step()
        weights_id = ray.put((train_step, weights))
        for q in self.param_queues:
            q.put(weights_id)

    def set_weights(self, weights):
        aid, sid, weights = weights
        assert aid == self.aid, f'{aid} != {self.aid}'  # we fix aid for now
        if sid != self.sid:
            print(f"Agent({self.aid})'s strategy changes from ({self.sid}) to ({sid})")
            self.sid = sid
            self.identifier = self._idc.get_identifier(aid, sid)
        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        self.strategy.set_weights(weights, identifier=identifier)

    def get_weights(self, identifier=None, model_only=True):
        if identifier is None:
            identifier = self._idc.get_identifier(aid=self.aid, sid=self.sid)
        if model_only:
            weights = self.model.get_weights()
            weights = {f'{identifier}_model': weights}
        else:
            weights = self.strategy.get_weights(identifier=identifier)

        return WeightsTuple(self.aid, self.sid, weights)

    def get_stats(self):
        """ retrieve training stats for the recorder to record """
        train_step = self.strategy.get_train_step()
        return train_step, self.recorder.get_stats()

    def get_buffer(self):
        return self.buffer


def create_remote_trainer(config, env_stats, name=None):
    ray_config = config.coordinator.trainer_ray
    config = AttrDict2dict(config)
    env_stats = AttrDict2dict(env_stats)
    return RemoteTrainer.as_remote(**ray_config
        ).remote(config, env_stats, name=name)


if __name__ == '__main__':
    from env.func import get_env_stats

    import numpy as np
    import ray
    from utility.yaml_op import load_config
    config = load_config('distributed/apg/config.yaml')

    for v in config.values():
        if isinstance(v, dict):
            v['root_dir'] = config['root_dir']
            v['model_name'] = config['model_name']

    ray.init()

    env_stats = get_env_stats(config.env)

    trainer1 = create_remote_trainer(config, env_stats, 0)
    trainer2 = create_remote_trainer(config, env_stats, 1)
    config = AttrDict2dict(config)
    ray.get(trainer1.construct_strategy_from_config.remote(
        config, 0, 0))
    ray.get(trainer2.construct_strategy_from_config.remote(
        config, 0, 0))

    weights = trainer1.get_weights.remote()
    trainer2.set_weights.remote(weights)
    aid2, sid2, weights2 = ray.get(trainer2.get_weights.remote())
    aid1, sid1, weights1 = ray.get(weights)
    for k in weights1.keys():
        # if k.endswith('model'):
        w1 = weights1[k]
        w2 = weights2[k]
        for v1, v2 in zip(w1, w2):
            np.testing.assert_allclose(v1, v2)

    time.sleep(2)
    ray.shutdown()
