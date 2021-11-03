import time
import threading
from ray.util.queue import Queue

from core.dataset import create_dataset
from core.mixin import IdentifierConstructor
from core.monitor import create_monitor
from core.tf_config import *
from core.utils import save_config
from distributed.typing import WeightsTuple
from utility import pkg
from utility.timer import Every
from utility.typing import AttrDict
from utility.utils import AttrDict2dict, config_attr, dict2AttrDict
from distributed.remote.base import RayBase


class RemoteTrainer(RayBase):
    def __init__(self, config, env_stats, aid=None, sid=None, name=None):
        self.config = config_attr(self, config, filter_dict=False)
        self._env_stats = env_stats
        self._name = name
        self._idc = IdentifierConstructor()

        self.model_constructor = None
        self.loss_constructor = None
        self.trainer_constructor = None
        self.strategy_constructor = None

        self.aid = aid
        self.sid = sid
        self.identifier = None
        self.model = None
        self.trainer = None
        self.strategy = None

        self.buffer = None
        self.dataset = None

        self.construct_strategy_from_config(self.config, self.aid, self.sid)

        self.actors = []
        self.param_queues = []
        self.monitor = create_monitor(
            self.config.root_dir, self.config.model_name, name)
        # we defer all constructions to the run time

    """ Construction """
    def construct_strategy_from_config(self, 
            config, aid=None, sid=None, weights=None, buffer=None):
        """ Construct a strategy from config """
        if not isinstance(config, AttrDict):
            config = dict2AttrDict(config)
        algo = config.algorithm.split('-')[-1]
        self._setup_constructors(algo)
        self._construct_trainer(config, self._env_stats)
        self._construct_dataset(config, buffer)
        self._construct_strategy(config)

        self.identifier = self._idc.get_identifier(aid, sid)
        if weights is not None:
            self.strategy.set_weights(weights, aid, sid)
        
        save_config(
            self.config.root_dir, self.config.model_name, 
            config, f'{aid}-{sid}')

    def _setup_constructors(self, algo):
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
        algo = self._algorithm.split('-')[0]
        if buffer is None:
            buffer_constructor = pkg.import_module(
                'buffer', pkg=f'distributed.{algo}').create_buffer
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
            use_ray=True)

    def _construct_strategy(self, config):
        self.strategy = self.strategy_constructor(
            self.identifier, config.strategy, 
            self.trainer, dataset=self.dataset)

    """ Actor """
    def register_actor(self, actor):
        pq = Queue()
        actor.register_buffer.remote(self.aid, self.buffer)
        actor.register_handler.remote(param_queue=pq)
        self.actors.append(actor)
        self.param_queues.append(pq)

    """ Training """
    def start_training(self):
        self._train_thread = threading.Thread(
            target=self._train, daemon=True)
        self._train_thread.start()

    def _train(self):
        if not hasattr(self, 'buffer'):
            raise RuntimeError(f'No buffer has been associate to trainer')
        # waits for enough data to train
        while hasattr(self.dataset, 'good_to_learn') \
                and not self.dataset.good_to_learn():
            time.sleep(1)
        print(f'{self._name} starts learning...')

        to_record = Every(self.LOG_PERIOD, self.LOG_PERIOD)
        while True:
            stats = self.strategy.train_record()
            aux = self.buffer.get_aux_stats.remote()
            self.monitor.store(**stats)
            identifier = self._idc.get_identifier(aid=self.aid, sid=self.sid)
            train_step, weights = self.get_train_step_weights(identifier)
            weights.weights[f'{identifier}_aux'] = ray.get(aux)
            weights.weights.pop(f'{identifier}_opt')
            weights_id = ray.put((self.aid, train_step, weights))
            for q in self.param_queues:
                q.put(weights_id)
            if to_record(train_step):
                stats = self.monitor.get_stats()


    """ Get & Set """
    def set_weights(self, weights, aid=None, sid=None):
        assert aid == self.aid, f'{aid} != {self.aid}'  # we fix aid for now
        if sid != self.sid:
            print(f"Agent({self.aid})'s strategy changes from ({self.sid}) to ({sid})")
            self.sid = sid
            self.identifier = self._idc.get_identifier(aid, sid)
        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        self.strategy.set_weights(weights, identifier=identifier)

    def get_weights(self, identifier=None):
        if identifier is None:
            identifier = self._idc.get_identifier(aid=self.aid, sid=self.sid)
        weights = self.strategy.get_weights(identifier=identifier)
        return WeightsTuple(self.aid, self.sid, weights)

    def get_train_step_weights(self, identifier=None):
        if identifier is None:
            identifier = self._idc.get_identifier(aid=self.aid, sid=self.sid)
        train_step = self.strategy.get_train_step()
        weights = self.get_weights(identifier)
        return train_step, weights

    def get_stats(self):
        """ retrieve training stats for the monitor to record """
        train_step = self.strategy.get_train_step()
        return train_step, self.monitor.get_stats()

    def get_buffer(self):
        return self.buffer


def create_trainer(config, env_stats, aid=None, sid=None, name=None):
    return RemoteTrainer(config, env_stats, aid, sid, name)


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

    trainer1 = create_trainer(config, env_stats, aid=1, sid=1)
    trainer2 = create_trainer(config, env_stats, aid=1, sid=2)

    aid1, sid1, weights1 = trainer1.get_weights()
    trainer2.set_weights(weights1, aid1, sid1)
    aid2, sid2, weights2 = trainer2.get_weights()

    for k in weights1.keys():
        # if k.endswith('model'):
        w1 = weights1[k]
        w2 = weights2[k]
        for v1, v2 in zip(w1, w2):
            np.testing.assert_allclose(v1, v2)

    time.sleep(2)
    ray.shutdown()
