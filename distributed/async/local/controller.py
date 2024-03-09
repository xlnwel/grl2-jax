import time
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
    max_steps: int
  ):
    agent_manager.start_training()
    runner_manager.start_running()
    to_restart_runners = Every(
      self.config.restart_runners_period, 
      0 if self.config.restart_runners_period is None \
        else self._steps + self.config.restart_runners_period
    )
    to_eval = Every(self.config.eval_period, start=self._steps, final=max_steps)
    to_store = Every(self.config.store_period, start=self._steps, final=max_steps)
    eval_pids = []

    while self._steps < max_steps:
      self._preprocessing(to_eval, to_restart_runners, to_store, eval_pids)
      time.sleep(1)

      steps = runner_manager.get_total_steps()
      self._steps = sum(steps)
      # do_logging(f'finishing sampling: total steps={steps}')
      is_score_met = self._check_scores()
      if is_score_met:
        break

    status = "score_met" if is_score_met else "timeout"
    self._finish_iteration(eval_pids, status=status)

  """ Implementation for <pbt_train> """
  def _prepare_configs(self, n_runners: int, n_steps: int, iteration: int):
    configs = [dict2AttrDict(c, to_copy=True) for c in self.configs]
    runner_stats = ray.get(self.parameter_server.get_runner_stats.remote())
    assert self._iteration == runner_stats.iteration, (self._iteration, runner_stats.iteration)
    do_logging(runner_stats, prefix=f'Runner Stats at Iteration {self._iteration}', color='blue')
    self._log_stats(runner_stats, self._iteration)
    return configs
