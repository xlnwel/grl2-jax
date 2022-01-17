from typing import List
import numpy as np
import ray
from ray.util.queue import Queue

from .parameter_server import ParameterServer
from .runner import MultiAgentSimRunner
from core.monitor import Monitor
from core.typing import ModelPath
from utility.typing import AttrDict
from utility.utils import AttrDict2dict, batch_dicts


class RunnerManager:
    def __init__(
        self, 
        config: AttrDict, 
        ray_config: AttrDict={}, 
        param_queues: List[List[Queue]]=None,
        parameter_server: ParameterServer=None,
        monitor: Monitor=None
    ):
        self.config = config
        self.param_queues = param_queues
        self.parameter_server = parameter_server
        self.monitor = monitor
        self.RemoteRunner = MultiAgentSimRunner.as_remote(**ray_config)

    """ Runner Management """
    def build_runners(
        self, 
        configs: List[dict], 
        store_data: bool=True,
        evaluation: bool=False
    ):
        self.runners: List[MultiAgentSimRunner] = [
            self.RemoteRunner.remote(
                [AttrDict2dict(config) for config in configs], 
                store_data=store_data, 
                evaluation=evaluation, 
                param_queues=[pqs[i] for pqs in self.param_queues] if self.param_queues else None, 
                parameter_server=self.parameter_server,
                monitor=self.monitor) 
            for i in range(self.config.n_workers)]

    """ Running Routines """
    def random_run(self):
        ray.get([r.random_run.remote() for r in self.runners])

    def run(self, wait=True):
        ids = [r.run.remote() for r in self.runners]
        return self._wait(ids, wait=wait)

    def evaluate(self, total_episodes):
        """ Evaluation is problematic if self.runner.run does not end in a pass """
        n_eps = 0
        video_list = []
        rewards_list = []
        stats_list = []
        i = 0
        while n_eps < total_episodes:
            _, eps, video, rewards, stats = zip(
                *ray.get([r.evaluate.remote() for r in self.runners]))
            n_eps += sum(eps)
            video_list += video
            rewards_list += rewards
            stats_list += stats
            i += 1
        print('Total number of runs:', i)
        stats = batch_dicts(stats_list, np.concatenate)
        video = sum(video_list, [])
        rewards = sum(rewards_list, [])
        return stats, n_eps, video, rewards

    """ Running Setups """
    def set_active_model_paths(self, model_paths: List[ModelPath], wait=False):
        pid = ray.put(model_paths)
        ids = [r.set_active_model_paths.remote(pid) for r in self.runners]
        self._wait(ids, wait=wait)

    def set_current_model_paths(self, model_paths: List[ModelPath], wait=False):
        pid = ray.put(model_paths)
        ids = [r.set_current_model_paths.remote(pid) for r in self.runners]
        self._wait(ids, wait=wait)

    def register_handler(self, wait=True, **kwargs):
        ids = [r.register_handler.remote(**kwargs) for r in self.runners]
        return self._wait(ids, wait=wait)

    """ Implementations """
    def _wait(self, ids, wait=False):
        if wait:
            return ray.get(ids)
        else:
            return ids
