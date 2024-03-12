import os
import time
from typing import Dict
import ray

from core.log import do_logging
from core.typing import ModelPath, dict2AttrDict, decompose_model_name
from tools.timer import Every, timeit
from distributed.common.names import EXPLOITER_SUFFIX
from distributed.common.utils import find_latest_model
from distributed.common.typing import Status
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

    while self._steps < max_steps:
      eval_pids = self._preprocessing(periods, eval_pids)
      time.sleep(1)

      steps = runner_manager.get_total_steps()
      self._steps = sum(steps)
      # do_logging(f'finishing sampling: total steps={steps}')
      is_score_met = self._check_scores()
      if is_score_met:
        break
      if self.exploiter:
        model = self.current_models[0]
        main_model = ModelPath(model.root_dir, model.model_name.replace(EXPLOITER_SUFFIX, ''))
        basic_name, aid = decompose_model_name(main_model.model_name)[:2]
        path = os.path.join(model.root_dir, basic_name, f'a{aid}')
        latest_main = find_latest_model(path)
        if latest_main != main_model:
          do_logging(f'Latest main model has been changed from {main_model} to {latest_main}', color='blue')
          break

    status = Status.SCORE_MET if is_score_met else Status.TIMEOUT
    self._finish_iteration(eval_pids, status=status)

  """ Implementation for <pbt_train> """
  def _prepare_configs(self, n_runners: int, n_steps: int, iteration: int):
    configs = [dict2AttrDict(c, to_copy=True) for c in self.configs]
    runner_stats = ray.get(self.parameter_server.get_runner_stats.remote())
    assert self._iteration == runner_stats.iteration, (self._iteration, runner_stats.iteration)
    do_logging(runner_stats, prefix=f'Runner Stats at Iteration {self._iteration}', color='blue')
    self._log_stats(runner_stats, self._iteration)
    return configs
