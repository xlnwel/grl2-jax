import threading
import ray

from .parameter_server import ParameterServer
from .typing import ModelStats, ModelWeights
from core.elements.builder import ElementsBuilder
from core.elements.strategy import Strategy
from core.monitor import Monitor
from core.remote.base import RayBase
from utility.utils import AttrDict2dict


class Agent(RayBase):
    def __init__(self, 
                 config: dict, 
                 env_stats: dict, 
                 parameter_server: ParameterServer,
                 monitor: Monitor
                 ):
        super().__init__()
        self.aid = config['aid']
        self.parameter_server = parameter_server
        self.monitor = monitor

        self.builder = ElementsBuilder(
            config=config, 
            env_stats=env_stats, 
            incremental_version=True)
        self.config = self.builder.config
        elements = self.builder.build_training_strategy_from_scratch(build_monitor=False)
        self.strategy: Strategy = elements.strategy
        self.strategy.save()
        self.buffer = elements.buffer

        self.train_step = None

    """ Version Control """
    def get_version(self):
        return self.builder.get_version()

    def increase_version(self):
        self.builder.increase_version()
        model_path = self.builder.get_model_path()
        self.strategy.reset_model_path(model_path)
        return model_path

    """ Stats Retrieval """
    def get_config(self):
        return AttrDict2dict(self.config)

    """ Model Management """
    def get_model_path(self):
        return self.strategy.get_model_path()

    def set_model_weights(self, model_weights):
        self.strategy.reset_model_path(model_weights.model)
        self.strategy.set_weights(model_weights.weights)

    """ Communications with Parameter Server """
    def publish_strategy(self, wait=False):
        ids = self.parameter_server.add_strategy_from_path.remote(
            self.aid, self.get_model_path(), set_active=True)
        self._wait(ids, wait)

    def publish_weights(self, wait=False):
        model_weights = ModelWeights(
            self.get_model_path(), 
            self.strategy.get_weights(train_step=True, env_step=False))
        ids = self.parameter_server.update_strategy_weights.remote(
            self.aid, model_weights)
        self._wait(ids, wait)

    """ Training """
    def start_training(self):
        self._training_thread = threading.Thread(target=self._training, daemon=True)
        self._training_thread.start()

    def _training(self):
        self.publish_weights()
        while True:
            stats = self.strategy.train_record()
            self.publish_weights()
            self._send_train_stats(stats)

    def _send_train_stats(self, stats):
        model_stats = ModelStats(self.get_model_path(), stats)
        self.monitor.store_train_stats.remote(model_stats)

    """ Data Management """
    def merge_episode(self, train_step, episode, n):
        # print('merge', train_step, self.train_step, self.buffer.ready(), n, self.buffer.size(), self.buffer.max_size())
        if train_step != self.train_step:
            return False
        if self.buffer.ready():
            return True
        self.buffer.merge_episode(episode, n)
        return self.buffer.ready()
    
    def merge_data(self, data, n):
        self.buffer.merge_data(data, n)

    def is_buffer_ready(self):
        print(self.buffer.size(), self.buffer.max_size())
        return self.buffer.ready()

    """ Checkpoints """
    def save(self):
        self.strategy.save()

    """ Implementations """
    def _wait(self, ids, wait=False):
        if wait:
            return ray.get(ids)
        else:
            return ids
