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
    def __init__(
        self, 
        config: dict,
        parameter_server: ParameterServer
    ):
        super().__init__()
        self.config = dict2AttrDict(config)
        self.parameter_server = parameter_server

        self.monitors: Dict[ModelPath, ModelMonitor] = {}

        self._to_store: Dict[ModelPath, Every] = {}

        self.train_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self.n_train_steps_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self.env_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self.n_env_steps_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self.n_episodes: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self.n_episodes_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._dir = '/'.join([self.config.root_dir, self.config.model_name])
        self._path = '/'.join([self._dir, 'monitor.pkl'])
        self.restore()

    def store_train_stats(self, model_stats: ModelStats):
        model, stats = model_stats
        self.build_monitor(model)
        train_step = stats.pop('train_step')
        self.n_train_steps_in_period[model] = train_step - self.train_steps[model]
        self.train_steps[model] = train_step
        self.monitors[model].store(**stats)

    def store_run_stats(self, model_stats: ModelStats):
        model, stats = model_stats
        self.build_monitor(model)
        env_steps = stats.pop('env_steps')
        self.env_steps[model] += env_steps
        self.n_env_steps_in_period[model] += env_steps
        n_episodes = stats.pop('n_episodes')
        self.n_episodes[model] += n_episodes
        self.n_episodes_in_period[model] += n_episodes

        self.monitors[model].store(**stats)

        if model not in self._to_store:
            self._to_store[model] = Every(self.config.store_period, time())
        if self._to_store[model](time()):
            stats = ray.get(self.parameter_server.get_aux_stats.remote(model))
            self.monitors[model].store(**stats)
            pid = self.parameter_server.save_active_model.remote(
                model, self.train_steps[model], self.env_steps[model])
            self.save()
            ray.get(pid)
            # we ensure that all checkpoint stats are saved to the disk 
            # before recording running/training stats. 
            self.record(model, self._to_store[model].difference())

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
        if not os.path.isdir(self._dir):
            os.makedirs(self._dir)
        with open(self._path, 'wb') as f:
            cloudpickle.dump([self.env_steps, self.train_steps, self.n_episodes], f)

    def record(self, model, duration):
        self.monitors[model].store(
            tps=self.n_train_steps_in_period[model] / duration,
            fps=self.n_env_steps_in_period[model] / duration,
            eps=self.n_episodes_in_period[model] / duration,
            train_step=self.train_steps[model],
            n_episodes=self.n_episodes[model],
        )
        self.monitors[model].record(self.env_steps[model])
        self.n_train_steps_in_period[model] = 0
        self.n_env_steps_in_period[model] = 0
        self.n_episodes_in_period[model] = 0
