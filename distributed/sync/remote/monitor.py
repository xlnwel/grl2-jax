import os
from datetime import datetime
import collections
import time
from typing import Any, Dict
import cloudpickle
import numpy as np
import ray

from .parameter_server import ParameterServer
from ..common.typing import ModelStats
from core.monitor import Monitor as ModelMonitor
from core.remote.base import RayBase
from core.typing import ModelPath
from utility.graph import get_tick_labels
from utility.timer import Timer
from utility.utils import dict2AttrDict


class Monitor(RayBase):
    def __init__(
        self, 
        config: dict,
        parameter_server: ParameterServer
    ):
        super().__init__(seed=config.get('seed'))
        self.config = dict2AttrDict(config['monitor'])
        self.n_agents = config['n_agents']
        self.parameter_server = parameter_server

        self._monitor = None
        self.monitors: Dict[ModelPath, ModelMonitor] = {}
        self._recording_stats: Dict[Dict[ModelPath, Any]] = collections.defaultdict(dict)

        self._train_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._train_steps_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._env_steps: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._env_steps_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._episodes: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)
        self._episodes_in_period: Dict[ModelPath, int] = collections.defaultdict(lambda: 0)

        self._last_save_time = time.time()

        self._dir = '/'.join([self.config.root_dir, self.config.model_name])
        self._path = '/'.join([self._dir, 'monitor.pkl'])

        self.restore()

    def build_monitor(self, model_path: ModelPath):
        if model_path not in self.monitors:
            self.monitors[model_path] = ModelMonitor(
                model_path, name=model_path.model_name)

    """ Stats Management """
    def store_stats(
        self, 
        stats: Dict, 
        step: int, 
        record=False
    ):
        if self._monitor is None:
            self._monitor = ModelMonitor(
                ModelPath(self.config.root_dir, self.config.model_name), 
                name=self.config.model_name
            )
        self._monitor.store(**stats)
        self._monitor.set_step(step)
        if record:
            self._monitor.record(step, print_terminal_info=True)

    def store_train_stats(
        self, 
        model_stats: ModelStats
    ):
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

    def record(
        self, 
        model: ModelPath, 
        duration: float, 
    ):
        n_items = 0
        size = 0
        for d in self._recording_stats.values():
            n_items += len(d)
            for v in d.values():
                size += v.nbytes
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

    def clear_iteration_stats(self):
        self._recording_stats.clear()
        self.monitors.clear()

    """ Checkpoints """
    def restore(self):
        if os.path.exists(self._path):
            with open(self._path, 'rb') as f:
                self._train_steps, \
                self._train_steps_in_period, \
                self._env_steps, \
                self._env_steps_in_period, \
                self._episodes, \
                self._episodes_in_period, \
                self._recording_stats = cloudpickle.load(f)

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
                self._recording_stats
            ), f)

    def save_all(self):
        def store_stats(model, stats, pids):
            assert model is not None, model
            self.build_monitor(model)
            self.monitors[model].store(**stats)
            pids.append(self.parameter_server.save_active_model.remote(
                model, self._train_steps[model], self._env_steps[model]
            ))

            with Timer('Monitor Recording Time', period=1):
                self.record(model, time.time() - self._last_save_time)

        pids = []
        oid = self.parameter_server.get_active_aux_stats.remote()
        self.save()
        if self.n_agents != 2:
            active_stats = ray.get(oid)
            for model, stats in active_stats.items():
                store_stats(model, stats, pids)
        else:
            active_stats, dists = ray.get([
                oid, 
                self.parameter_server.get_opponent_distributions_for_active_models.remote()
            ])
            for model, (payoff, dist) in dists.items():
                store_stats(model, active_stats[model], pids)

                with Timer('Monitor Real-Time Plot Time', period=1):
                    self.save_real_time_opp_dist(model, payoff, dist)

        ray.get(pids)
        self._last_save_time = time.time()
        with open('check.txt', 'w') as f:
            f.write(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))

    def save_payoff_table(self, step=None):
        if self.n_agents == 2:
            with Timer('Monitor Retrieval Time', period=1):
                models, payoffs, counts = ray.get([
                    self.parameter_server.get_active_models.remote(), 
                    self.parameter_server.get_payoffs.remote(), 
                    self.parameter_server.get_counts.remote()
                ])
            with Timer('Monitor Matrix Plot Time', period=1):
                for m, p, c in zip(models, payoffs, counts):
                    self.save_payoff_table_for_model(m, p, c, step=step)

    def save_real_time_opp_dist(
        self,
        model: ModelPath, 
        payoff: np.ndarray, 
        dist: np.ndarray, 
    ):
        payoff = np.reshape(payoff, (-1, 1)).astype(np.float16)
        dist = np.reshape(dist, (-1, 1)).astype(np.float16)
        rp = self._recording_stats['payoffs']
        rod = self._recording_stats['opp_dists']
        if model in rod:
            rp[model] = np.concatenate([rp[model], payoff], -1)
            rod[model] = np.concatenate([rod[model], dist], -1)
        else:
            rp[model] = payoff
            rod[model] = dist

        rp[model][np.isnan(rp[model])] = np.nanmin(rp[model])
        xlabel = 'Step'
        ylabel = 'Opponent'
        xticklabels = get_tick_labels(rod[model].shape[1])
        yticklabels = get_tick_labels(rod[model].shape[0])
        self.monitors[model].matrix_summary(
            model=model, 
            matrix=rp[model], 
            xlabel=xlabel, 
            ylabel=ylabel, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels, 
            name='realtime_payoff', 
            step=self._env_steps[model]
        )
        self.monitors[model].matrix_summary(
            model=model, 
            matrix=rod[model], 
            xlabel=xlabel, 
            ylabel=ylabel, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels, 
            name='realtime_opp_dists', 
            step=self._env_steps[model]
        )

    def save_payoff_table_for_model(
        self, 
        model: ModelPath, 
        payoff: np.ndarray, 
        counts: np.ndarray=None,
        step=None, 
    ):
        if step is None:
            step = self._env_steps[model]
        xlabel='Player2'
        ylabel='Player1'
        xticklabels = get_tick_labels(payoff.shape[1])
        yticklabels = get_tick_labels(payoff.shape[0])
        self.monitors[model].matrix_summary(
            model=model, 
            matrix=payoff, 
            xlabel=xlabel, 
            ylabel=ylabel, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels, 
            name='payoff', 
            step=step
        )
        self.monitors[model].matrix_summary(
            model=model, 
            matrix=counts, 
            xlabel=xlabel, 
            ylabel=ylabel, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels, 
            name='count', 
            step=step
        )
