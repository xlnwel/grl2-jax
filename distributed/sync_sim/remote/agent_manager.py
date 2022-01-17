from typing import List
import ray

from .agent import Agent
from .parameter_server import ParameterServer
from .typing import ModelWeights
from core.monitor import Monitor
from utility.typing import AttrDict
from utility.utils import AttrDict2dict


class AgentManager:
    def __init__(
        self, 
        ray_config: AttrDict, 
        env_stats: AttrDict, 
        parameter_server: ParameterServer,
        monitor: Monitor
    ):
        self.RemoteAgent = Agent.as_remote(**ray_config)
        self.env_stats = AttrDict2dict(env_stats)
        self.parameter_server = parameter_server
        self.monitor = monitor

    """ Agent Management """
    def build_agents(self, configs):
        self.agents: List[Agent] = [self.RemoteAgent.remote(
            config=AttrDict2dict(config),
            env_stats=self.env_stats,
            parameter_server=self.parameter_server,
            monitor=self.monitor,
        ) for config in configs]

    def destroy_agents(self):
        del self.agents

    """ Model Management """
    def get_model_paths(self, wait=True):
        ids = [a.get_model_path.remote() for a in self.agents]
        return self._wait(ids, wait)

    def set_model_weights(self, model_weights: ModelWeights, wait=False):
        ids = [a.set_model_weights.remote(mw) 
            for a, mw in zip(self.agents, model_weights)]
        self._wait(ids, wait)

    """ Communications with Parameter Server """
    def publish_weights(self, wait=False):
        ids = [a.publish_weights.remote(wait=wait) for a in self.agents]
        self._wait(ids, wait)

    """ Get """
    def get_agents(self):
        return self.agents

    """ Training """
    def start_training(self, wait=False):
        ids = [a.start_training.remote() for a in self.agents]
        self._wait(ids, wait)

    # """ Checkpoints """
    # def save(self, wait=False):
    #     ids = [a.save.remote() for a in self.agents]
    #     self._wait(ids, wait)

    """ Implementations """
    def _wait(self, ids, wait=False):
        if wait:
            return ray.get(ids)
        else:
            return ids
