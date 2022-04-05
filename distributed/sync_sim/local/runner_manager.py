from math import ceil
from typing import List, Union
import numpy as np
import ray

from ..remote.parameter_server import ParameterServer
from ..remote.runner import MultiAgentSimRunner
from core.monitor import Monitor
from core.remote.base import ManagerBase, RayBase
from core.typing import ModelPath
from utility.display import pwt
from utility.typing import AttrDict
from utility.utils import AttrDict2dict, batch_dicts


class RunnerManager(ManagerBase):
    def __init__(
        self, 
        config: AttrDict, 
        ray_config: AttrDict={}, 
        parameter_server: ParameterServer=None,
        monitor: Monitor=None
    ):
        self.config = config
        self.parameter_server = parameter_server
        self.monitor = monitor
        self.RemoteRunner = MultiAgentSimRunner.as_remote(**ray_config)

    """ Runner Management """
    def build_runners(
        self, 
        configs: Union[List[AttrDict], AttrDict], 
        remote_buffers: List[RayBase]=None, 
        active_models: List[ModelPath]=None, 
        store_data: bool=True,
        evaluation: bool=False
    ):
        if isinstance(configs, list):
            configs = [AttrDict2dict(config) for config in configs]
        else:
            configs = AttrDict2dict(configs)
        self.runners: List[MultiAgentSimRunner] = [
            self.RemoteRunner.remote(
                i,
                configs, 
                store_data=store_data, 
                evaluation=evaluation, 
                parameter_server=self.parameter_server, 
                remote_buffers=remote_buffers, 
                active_models=active_models, 
                monitor=self.monitor)
            for i in range(self.config.n_runners)
        ]
    
    @property
    def n_runners(self):
        return len(self.runners) if hasattr(self, 'runners') else 0

    def destroy_runners(self):
        del self.runners

    """ Running Routines """
    def random_run(self, aids=None, wait=False):
        self._remote_call_with_value(self.runners, 'random_run', aids, wait)

    def run_with_model_weights(self, mids, wait=True):
        oids = [runner.run_with_model_weights.remote(mid) 
            for runner, mid in zip(self.runners, mids)]
        return self._wait(oids, wait=wait)

    def run(self, wait=True):
        return self._remote_call(self.runners, 'run', wait=wait)

    def evaluate_all(self, total_episodes):
        strategies = ray.get(
            self.parameter_server.sample_strategies_for_evaluation.remote()
        )

        pwt('The total number of strategy tuples:', len(strategies))
        for s in strategies:
            self.set_weights_from_model_paths(s)
            self.evaluate(total_episodes)

    def evaluate(self, total_episodes):
        """ Evaluation is problematic if self.runner.run does not end in a pass """
        episodes_per_runner = ceil(total_episodes / len(self.runners))
        oid = ray.put(episodes_per_runner)
        steps, n_episodes = zip(
            *ray.get([r.evaluate.remote(oid) for r in self.runners])
        )

        steps = sum(steps)
        n_episodes = sum(n_episodes)

        return steps, n_episodes

    def evaluate_and_return_stats(self, total_episodes):
        """ Evaluation is problematic if self.runner.run does not end in a pass """
        episodes_per_runner = ceil(total_episodes / len(self.runners))
        oid = ray.put(episodes_per_runner)
        steps, n_episodes, videos, rewards, stats = zip(
            *ray.get([r.evaluate_and_return_stats.remote(oid) for r in self.runners]))

        steps = sum(steps)
        n_episodes = sum(n_episodes)
        videos = sum(videos, [])
        rewards = sum(rewards, [])
        stats = batch_dicts(stats, np.concatenate)
        return steps, n_episodes, videos, rewards, stats

    """ Running Setups """
    def set_active_models(self, model_paths: List[ModelPath], wait=False):
        self._remote_call_with_value(
            self.runners, 'set_active_models', model_paths, wait)

    def set_current_models(self, model_paths: List[ModelPath], wait=False):
        self._remote_call_with_value(
            self.runners, 'set_current_models', model_paths, wait)
    
    def set_weights_from_configs(self, configs: List[dict], wait=False):
        configs = [AttrDict2dict(c) for c in configs]
        self._remote_call_with_value(
            self.runners, 'set_weights_from_configs', configs, wait)

    def set_weights_from_model_paths(self, models: List[ModelPath], wait=False):
        self._remote_call_with_value(
            self.runners, 'set_weights_from_model_paths', models, wait)

    def set_running_steps(self, n_steps, wait=False):
        self._remote_call_with_value(
            self.runners, 'set_running_steps', n_steps, wait)

    """ Hanlder Registration """
    def register_handler(self, wait=True, **kwargs):
        self._remote_call_with_args(
            self.runners, 'register_handler', wait=wait, **kwargs)
