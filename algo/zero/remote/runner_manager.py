from typing import List, Union
import ray
from ray._raylet import ObjectRef
from ray.util.queue import Queue

from .agent import Agent
from .parameter_server import ParameterServer
from .runner import MultiAgentSimRunner
from core.monitor import Monitor
from core.typing import ModelPath
from utility.typing import AttrDict
from utility.utils import AttrDict2dict, batch_dicts


class RunnerManager:
    def __init__(self, 
                 config: AttrDict, 
                 ray_config: AttrDict, 
                 store_data=True, 
                 remote_agents: List[Agent]=None, 
                 param_queues: List[List[Queue]]=None,
                 parameter_server: ParameterServer=None,
                 monitor: Monitor=None
                 ):
        self.config = config
        self.store_data = store_data
        self.remote_agents = remote_agents
        self.param_queues = param_queues
        self.parameter_server = parameter_server
        self.monitor = monitor
        self.RemoteRunner = MultiAgentSimRunner.as_remote(**ray_config)

    """ Runner Management """
    def build_runners(self, 
                      configs: List[Union[dict, ObjectRef]], 
                      evaluation: bool=False):
        self.runners: List[MultiAgentSimRunner] = [
            self.RemoteRunner.remote(
                [AttrDict2dict(config) for config in configs], 
                store_data=self.store_data, 
                evaluation=evaluation, 
                remote_agents=self.remote_agents, 
                param_queues=[pqs[i] for pqs in self.param_queues], 
                parameter_server=self.parameter_server,
                monitor=self.monitor) 
            for i in range(self.config.n_workers)]

    """ Running Routines """
    def random_run(self):
        ray.get([r.random_run.remote() for r in self.runners])

    def run(self, wait=True):
        ids = [r.run.remote() for r in self.runners]
        self._wait(ids, wait=wait)
        
    def evaluate(self, total_episodes, weights=None):
        """ Evaluation is problematic if self.runner.run does not end in a pass """
        n_eps = 0
        stats_list = []
        i = 0
        while n_eps < total_episodes:
            _, _, stats = self.run(weights)
            n_eps += len(next(iter(stats.values())))
            stats_list.append(stats)
            i += 1
        print('Total number of runs:', i)
        stats = batch_dicts(stats_list, lambda x: sum(x, []))
        return stats, n_eps

    """ Running Setups """
    def set_active_model_paths(self, model_paths: List[ModelPath], wait=False):
        pid = ray.put(model_paths)
        ids = [r.set_active_model_paths.remote(pid) for r in self.runners]
        self._wait(ids, wait=wait)

    """ Implementations """
    def _wait(self, ids, wait=False):
        if wait:
            return ray.get(ids)
        else:
            return ids
