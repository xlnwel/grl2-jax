import os 
import collections
import cloudpickle
from time import time
from typing import Dict
import ray

from .parameter_server import ParameterServer
from .typing import ModelStats
from core.monitor import Monitor as ModelMonitor
from core.remote.base import RayBase
from core.typing import ModelPath
from utility.timer import Every
from utility.utils import dict2AttrDict


class Monitor(RayBase):
    def __init__(self, 
                 config: dict,
                 parameter_server: ParameterServer):
        super().__init__()
        self.config = dict2AttrDict(config)
        self.parameter_server = parameter_server

        self.monitors: Dict[ModelPath, ModelMonitor] = {}

        self._to_store: Dict[ModelPath, Every] = {}

        self.train_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self.env_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self.n_episodes: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._path = '/'.join([self.config.root_dir, self.config.model_name, 'monitor.pkl'])

    def store_train_stats(self, model_stats: ModelStats):
        self.build_monitor(model_stats.model)
        self.monitors[model_stats.model].store(**model_stats.stats)

    def store_run_stats(self, model_stats: ModelStats):
        model = model_stats.model
        stats = model_stats.stats
        self.build_monitor(model)
        self.train_steps[model] += stats.pop('train_step')
        self.env_steps[model] += stats.pop('env_step')
        self.n_episodes[model] += stats.pop('n_episodes')
        self.monitors[model].store(**stats)

        if model not in self._to_store:
            self._to_store[model] = Every(self.config.store_period, time())
        if self._to_store[model](time()):
            stats = ray.get(self.parameter_server.get_aux_stats.remote(model))
            self.monitors[model].store(**stats)
            self.record(model)
            self.save()
            self.parameter_server.save_active_models.remote(
                self.train_steps[model], self.env_steps[model])

    def build_monitor(self, model_path):
        if model_path not in self.monitors:
            self.monitors[model_path] = ModelMonitor(
                model_path, name=model_path.model_name)

    """ Save & Restore """
    def restore(self):
        if os.path.exists(self._path):
            with open(self._path, 'rb') as f:
                self.env_steps, self.train_steps, self.n_episodes = cloudpickle.load(f)

    def save(self):
        with open(self._path, 'wb') as f:
            cloudpickle.dump([self.env_steps, self.train_steps, self.n_episodes], f)

    def record(self, model):
        if self.monitors[model].contains_stats('train_step'):
            train_step = self.monitors[model].get_raw_item('train_step')[-1]
            self.monitors[model].store(train_step=train_step)
        self.monitors[model].store(n_episodes=self.n_episodes[model])
        self.monitors[model].record(self.env_steps[model])
