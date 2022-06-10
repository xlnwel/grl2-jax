import os
from datetime import datetime
import collections
import sys
from time import time
from typing import Dict
import cloudpickle
import numpy as np
import ray

from utility.graph import decode_png, matrix_plot

from .parameter_server import ParameterServer
from ..common.typing import ModelStats
from core.monitor import Monitor as ModelMonitor
from core.remote.base import RayBase
from core.typing import ModelPath, get_aid
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

        self._monitor = ModelMonitor(
            ModelPath(self.config.root_dir, self.config.model_name), 
            name=self.config.model_name
        )
        self.monitors: Dict[ModelPath, ModelMonitor] = {}
        self._opp_dists: Dict[ModelPath, np.ndarray] = {}
        self._payoffs: Dict[ModelPath, np.ndarray] = {}

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
    def store_stats(self, stats, step, record=False):
        self._monitor.store(**stats)
        self._monitor.set_step(step)
        if record:
            self._monitor.record(step, print_terminal_info=True)

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
        dists = ray.get(
            self.parameter_server.get_opponent_distributions_for_active_models.remote()
        )

        assert all([m is not None for m in dists]), dists
        current_time = time()
        for model, (payoff, dist) in dists.items():
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
            with Timer('Monitor Real-Time Plot Time', period=1):
                if len(dists) == 2:
                    dist = np.reshape(dist, (-1, 1))
                    payoff = np.reshape(payoff, (-1, 1))
                    if model in self._opp_dists:
                        self._opp_dists[model] = np.concatenate(
                            [self._opp_dists[model], dist], -1)
                        self._payoffs[model] = np.concatenate(
                            [self._payoffs[model], payoff], -1)
                    else:
                        self._opp_dists[model] = dist
                        self._payoffs[model] = payoff
                    aid = get_aid(model.model_name)
                    self._plot_matrix(
                        model=model, 
                        matrix=self._opp_dists[model], 
                        xlabel='Steps', 
                        ylabel='Opponent', 
                        xticklabels=self._opp_dists[model].shape[1] // 10 if self._opp_dists[model].shape[1] > 10 else 'auto', 
                        yticklabels=range(1, self._opp_dists[model].shape[0] + 1), 
                        name=f'opp_dists{aid}', 
                        step=0
                    )
                    self._plot_matrix(
                        model=model, 
                        matrix=self._payoffs[model], 
                        xlabel='Steps', 
                        ylabel='Opponent', 
                        xticklabels=self._payoffs[model].shape[1] // 10 if self._payoffs[model].shape[1] > 10 else 'auto', 
                        yticklabels=range(1, self._payoffs[model].shape[0] + 1), 
                        name=f'realtime_payoff{aid}', 
                        step=0
                    )
        self._last_save_time = current_time
        with open('check.txt', 'w') as f:
            f.write(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))

    def save_payoff_table(self, step=None):
        with Timer('Monitor Retrieval Time', period=1):
            mid = self.parameter_server.get_active_models.remote()
            pid = self.parameter_server.get_payoffs.remote()
            cid = self.parameter_server.get_counts.remote()
            models, payoffs, counts = ray.get([mid, pid, cid])

        if len(payoffs) == 2:
            with Timer('Monitor Matrix Plot Time', period=1):
                for i, (m, p, c) in enumerate(zip(models, payoffs, counts)):
                    if step is None:
                        step = self._env_steps[m]
                    xticklabels = range(1, p.shape[1] + 1)
                    yticklabels = range(1, p.shape[0] + 1)
                    self._plot_matrix(
                        model=m, 
                        matrix=p, 
                        xlabel='Player2', 
                        ylabel='Player1', 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels, 
                        name=f'payoff{i}', 
                        step=step
                    )
                    self._plot_matrix(
                        model=m, 
                        matrix=c, 
                        xlabel='Player2', 
                        ylabel='Player1', 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels, 
                        name=f'count{i}', 
                        step=step
                    )


    def _plot_matrix(
        self, 
        *, 
        model, 
        matrix, 
        label_top=True, 
        label_bottom=False, 
        xlabel, 
        ylabel, 
        xticklabels, 
        yticklabels,
        name, 
        step, 
    ):
        buf = matrix_plot(
            matrix, 
            label_top=label_top, 
            label_bottom=label_bottom, 
            save_path='/'.join([*model, name]), 
            xlabel=xlabel, 
            ylabel=ylabel, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels
        )
        count = decode_png(buf.getvalue())
        self.monitors[model].image_summary(
            name, count, step=step)