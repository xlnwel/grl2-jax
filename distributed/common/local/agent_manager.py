from typing import List
import ray

from core.elements.monitor import Monitor
from distributed.common.remote.base import ManagerBase
from core.typing import AttrDict, AttrDict2dict
from core.typing import ModelWeights
from ..remote.agent import Agent
from ..remote.parameter_server import ParameterServer


class AgentManager(ManagerBase):
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
    self.agents = None
    self.models = []

  """ Agent Management """
  def build_agents(self, configs: List[dict]):
    if self.agents:
      return
    self.agents: List[Agent] = [self.RemoteAgent.remote(
      config=config,
      env_stats=self.env_stats,
      parameter_server=self.parameter_server,
      monitor=self.monitor,
    ) for config in configs]

  def get_agents(self):
    return self.agents

  def destroy_agents(self):
    for a in self.agents:
      ray.kill(a)
    self.agents = None

  """ Model Management """
  def get_model_paths(self, wait=True):
    return self._remote_call(self.agents, 'get_model_paths', wait=wait)

  def set_model_weights(self, model_weights: List[ModelWeights], wait=False):
    self.models = [m.model for m in model_weights]
    self._remote_call_with_list(
      self.agents, 'set_model_weights', model_weights, wait=wait)

  """ Communications with Parameter Server """
  def publish_weights(self, wait=False):
    self._remote_call(self.agents, 'publish_weights', wait=wait)

  """ Training """
  def start_training(self, wait=False):
    self._remote_call(self.agents, 'start_training', wait=wait)

  def stop_training(self, wait=False):
    self._remote_call(self.agents, 'stop_training', wait=wait)

  def train(self, wait=True):
    return self._remote_call(self.agents, 'train', wait=wait)

  """ Hanlder Registration """
  def register_handler(self, wait=True, **kwargs):
    self._remote_call_with_args(
      self.agents, 'register_handler', wait=wait, **kwargs)
