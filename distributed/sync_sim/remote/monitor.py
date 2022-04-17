import os
from datetime import datetime
import collections
import sys
from time import time
from typing import Dict
import cloudpickle
import ray

from .parameter_server import ParameterServer
from ..common.typing import ModelStats
from core.monitor import Monitor as ModelMonitor
from core.remote.base import RayBase
from core.typing import ModelPath
from utility.timer import Timer
from utility.utils import dict2AttrDict


class Monitor(RayBase):
    def __init__(
        self, 
        config: dict,
        parameter_server: ParameterServer
    ):
        super().__init__(seed=config.get('seed'))
        self.config = dict2AttrDict(config)
        self.parameter_server = parameter_server

        self.monitors: Dict[ModelPath, ModelMonitor] = {}

        self._train_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._train_steps_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._env_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._env_steps_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._episodes: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._episodes_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)

        self._last_save_time = time()

        self._dir = '/'.join([self.config.root_dir, self.config.model_name])
        self._path = '/'.join([self._dir, 'monitor.pkl'])

        self.restore()

    def build_monitor(self, model_path: ModelPath):
        if model_path not in self.monitors:
            self.monitors[model_path] = ModelMonitor(
                model_path, name=model_path.model_name)

    """ Stats Storing """
    def store_train_stats(self, model_stats: ModelStats):
        model, stats = model_stats
        self.build_monitor(model)
        train_step = stats.pop('train_step')
        self._train_steps_in_period[model] = train_step - self._train_steps[model]
        self._train_steps[model] = train_step
        self.monitors[model].store(**stats)

    def store_run_stats(self, model_stats: ModelStats):
        model, stats = model_stats
        self.build_monitor(model)
        env_steps = stats.pop('env_steps')
        self._env_steps[model] += env_steps
        self._env_steps_in_period[model] += env_steps
        n_episodes = stats.pop('n_episodes')
        self._episodes[model] += n_episodes
        self._episodes_in_period[model] += n_episodes

        self.monitors[model].store(**{
            k if k.endswith('score') or '/' in k else f'run/{k}': v
            for k, v in stats.items()
        })

    def record(self, model, duration):
        self.monitors[model].store(**{
            'time/tps': self._train_steps_in_period[model] / duration,
            'time/fps': self._env_steps_in_period[model] / duration,
            'time/eps': self._episodes_in_period[model] / duration,
            'stats/train_step': self._train_steps[model],
            'run/n_episodes': self._episodes[model],
        })
        self._train_steps_in_period[model] = 0
        self._env_steps_in_period[model] = 0
        self._episodes_in_period[model] = 0
        self.monitors[model].record(self._env_steps[model], print_terminal_info=True)

    """ Checkpoints """
    def restore(self):
        if os.path.exists(self._path):
            with open(self._path, 'rb') as f:
                self._train_steps, \
                self._train_steps_in_period, \
                self._env_steps, \
                self._env_steps_in_period, \
                self._episodes, \
                self._episodes_in_period = cloudpickle.load(f)

    def save(self):
        if not os.path.isdir(self._dir):
            os.makedirs(self._dir)
        with open(self._path, 'wb') as f:
            cloudpickle.dump((
                self._train_steps, 
                self._train_steps_in_period, 
                self._env_steps, 
                self._env_steps_in_period,
                self._episodes, 
                self._episodes_in_period,
            ), f)

    def save_all(self):
        models = ray.get(self.parameter_server.get_active_models.remote())
        assert all([m is not None for m in models]), models
        current_time = time()
        for model in models:
            stats = ray.get(self.parameter_server.get_aux_stats.remote(model))
            self.build_monitor(model)
            self.monitors[model].store(**stats)
            self.parameter_server.save_active_model.remote(
                model, self._train_steps[model], self._env_steps[model])
            self.save()
            # we ensure that all checkpoint stats are saved to the disk 
            # before recording running/training stats. 
            with Timer('Monitor Recording Time', period=1):
                self.record(model, current_time - self._last_save_time)
        sys.stdout.flush()
        self._last_save_time = current_time
        with open('check.txt', 'w') as f:
            f.write(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))
