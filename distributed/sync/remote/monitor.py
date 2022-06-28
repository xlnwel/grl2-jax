import os
from datetime import datetime
import collections
import time
from typing import Any, Dict
import cloudpickle
import numpy as np
import ray

from rule.utils import is_rule_strategy

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
    def store_stats_for_model(
        self, 
        model: ModelPath, 
        stats: Dict, 
        step: int=None, 
        record=False, 
    ):
        self.build_monitor(model)
        self.monitors[model].store(**stats)
        if record:
            self.record_for_model(model, step)

    def store_stats(
        self, 
        *, 
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
        train_step = stats.pop('train_step')
        self._train_steps_in_period[model] = train_step - self._train_steps[model]
        self._train_steps[model] = train_step
        self.store_stats_for_model(model, stats)

    def store_run_stats(self, model_stats: ModelStats):
        model, stats = model_stats
        self.build_monitor(model)
        env_steps = stats.pop('env_steps')
        self._env_steps[model] += env_steps
        self._env_steps_in_period[model] += env_steps
        n_episodes = stats.pop('n_episodes')
        self._episodes[model] += n_episodes
        self._episodes_in_period[model] += n_episodes

        stats = {
            k if k.endswith('score') or '/' in k else f'run/{k}': v
            for k, v in stats.items()
        }
        self.store_stats_for_model(
            model, 
            stats, 
            record=is_rule_strategy(model), 
        )

    def record_for_model(
        self, 
        model: ModelPath, 
        step, 
    ):
        n_items = 0
        size = 0
        for d in self._recording_stats.values():
            n_items += len(d)
            for v in d.values():
                size += v.nbytes
        duration = time.time() - self._last_save_time
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
        self.monitors[model].set_step(step)
        self.monitors[model].record(print_terminal_info=True)
        self._last_save_time = time.time()

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

    def save_all(self, step):
        def store_stats(model, stats, pids):
            assert model is not None, model
            pids.append(self.parameter_server.save_active_model.remote(
                model, self._train_steps[model], self._env_steps[model]
            ))
            self.store_stats_for_model(model, stats, step=step, record=True)

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
                    self.plot_recording_stats(model, 'payoff', payoff, fill_nan=True)
                    self.plot_recording_stats(model, 'opp_dist', dist)

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
                    stats = {}
                    for i, (pp, cc) in enumerate(zip(p, c)):
                        stats[f'payoff{i}'] = pp
                        stats[f'count{i}'] = cc

                    self.monitors[m].store(payoff=p, count=c)
                    self.plot_stats(
                        model=m,
                        stats=p, 
                        xlabel='Player2', 
                        ylabel='Player1', 
                        name='payoff', 
                        step=step
                    )
                    self.plot_stats(
                        model=m,
                        stats=c, 
                        xlabel='Player2', 
                        ylabel='Player1', 
                        name='count', 
                        step=step
                    )

    def plot_recording_stats(
        self, 
        model, 
        name, 
        stats, 
        fill_nan=False
    ):
        self.update_recording_stats(
            model, name, stats, fill_nan=fill_nan
        )
        self.plot_stats(
            model, 
            stats=self._recording_stats[name][model], 
            xlabel='Step', 
            ylabel='Opponent', 
            name=f'realtime_{name}', 
        )

    def update_recording_stats(
        self, 
        model: ModelPath, 
        stats_name: str, 
        new_stats: np.ndarray, 
        fill_nan: bool=False
    ):
        new_stats = np.reshape(new_stats, (-1, 1)).astype(np.float16)
        hist_stats = self._recording_stats[stats_name]
        if model in hist_stats:
            hist_stats[model] = np.concatenate([hist_stats[model], new_stats], -1)
        else:
            hist_stats[model] = new_stats
        
        if fill_nan:
            hist_stats[model][np.isnan(hist_stats[model])] = np.nanmin(hist_stats[model])
        
        return hist_stats
        
    def plot_stats(
        self, 
        model: ModelPath, 
        stats: np.ndarray, 
        xlabel: str, 
        ylabel: str, 
        name: str, 
        step=None, 
    ):
        xticklabels = get_tick_labels(stats.shape[1])
        yticklabels = get_tick_labels(stats.shape[0])
        self.monitors[model].matrix_summary(
            model=model, 
            matrix=stats, 
            xlabel=xlabel, 
            ylabel=ylabel, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels, 
            name=name, 
            step=self._env_steps[model] if step is None else step
        )
