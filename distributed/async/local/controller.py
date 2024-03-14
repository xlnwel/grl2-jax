import os
import time
from typing import Dict
import ray

from core.log import do_logging
from core.typing import dict2AttrDict
from tools.timer import Every, timeit
from distributed.common.local.agent_manager import AgentManager
from distributed.common.local.runner_manager import RunnerManager
from distributed.common.local.controller import Controller as ControllerBase


class Controller(ControllerBase):
  @timeit
  def train(
    self, 
    agent_manager: AgentManager, 
    runner_manager: RunnerManager, 
    max_steps: int, 
    periods: Dict[str, Every]
  ):
    agent_manager.start_training()
    runner_manager.start_running()
    eval_pids = []

    while True:
      eval_pids = self._preprocessing(periods, eval_pids)
      time.sleep(1)

      steps = runner_manager.get_total_steps()
      self._steps = sum(steps)

      if self._check_termination(self._steps, max_steps):
        break

    self._finish_iteration(eval_pids)

  """ Implementation for <pbt_train> """
  def _prepare_configs(self, n_runners: int, n_steps: int, iteration: int):
    configs = [dict2AttrDict(c, to_copy=True) for c in self.configs]
    runner_stats = ray.get(self.parameter_server.get_runner_stats.remote())
    assert self._iteration == runner_stats.iteration, (self._iteration, runner_stats.iteration)
    do_logging(runner_stats, prefix=f'Runner Stats at Iteration {self._iteration}', color='blue')
    self._log_stats(runner_stats, self._iteration)
    return configs
