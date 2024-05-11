from math import ceil
from typing import List, Union
import numpy as np
import ray

from ..remote.parameter_server import ParameterServer
from ..remote.runner import MultiAgentRunner
from tools.log import do_logging
from core.elements.monitor import Monitor
from core.remote.base import ManagerBase, RayBase
from core.typing import ModelPath, AttrDict, AttrDict2dict
from tools.utils import batch_dicts


class RunnerManager(ManagerBase):
  def __init__(
    self, 
    ray_config: AttrDict={}, 
    parameter_server: ParameterServer=None,
    monitor: Monitor=None
  ):
    self.parameter_server = parameter_server
    self.monitor = monitor
    self.RemoteRunner = MultiAgentRunner.as_remote(**ray_config)
    self.runners = None

  """ Runner Management """
  def build_runners(
    self, 
    configs: Union[List[AttrDict], AttrDict], 
    remote_buffers: List[RayBase]=None, 
    active_models: List[ModelPath]=None, 
    evaluation: bool=False
  ):
    if self.runners:
      for r in self.runners:
        r.set_active_models.remote(active_models)
      return
    if isinstance(configs, list):
      config = configs[0]
      configs = [AttrDict2dict(config) for config in configs]
    else:
      config = configs
      configs = AttrDict2dict(configs)
    self.runners: List[MultiAgentRunner] = [
      self.RemoteRunner.remote(
        i,
        configs, 
        evaluation=evaluation, 
        parameter_server=self.parameter_server, 
        remote_buffers=remote_buffers, 
        active_models=active_models, 
        monitor=self.monitor)
      for i in range(config.runner.n_runners)
    ]
  
  @property
  def n_runners(self):
    return len(self.runners) if hasattr(self, 'runners') else 0

  def destroy_runners(self):
    for r in self.runners:
      ray.kill(r)
    self.runners = None

  def get_total_steps(self):
    return self._remote_call(self.runners, 'get_total_steps', wait=True)

  """ Running Routines """
  def random_run(self, aids=None, wait=False):
    self._remote_call_with_value(self.runners, 'random_run', aids, wait)

  def start_running(self, wait=False):
    self._remote_call(self.runners, 'start_running', wait=wait)
    
  def stop_running(self, wait=False):
    self._remote_call(self.runners, 'stop_running', wait=wait)

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

    do_logging(f'The total number of strategy tuples: {len(strategies)}', level='info')
    eids = []
    rid = 0
    for s in strategies:
      eids.append(self.runners[rid].evaluate.remote(total_episodes, s, wait=True))
      rid += 1
      if rid == len(self.runners):
        rid = 0
    assert len(eids) == len(strategies), (len(eids), len(strategies))
    ray.get(eids)

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
