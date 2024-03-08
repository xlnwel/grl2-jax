from math import ceil
import numpy as np
import ray

from core.log import do_logging
from core.typing import dict2AttrDict
from tools.timer import Every, timeit
from distributed.common.typing import Status
from distributed.common.local.agent_manager import AgentManager
from distributed.common.local.runner_manager import RunnerManager
from distributed.common.local.controller import Controller as ControllerBase


def _compute_pbt_steps(n_runners, n_steps, n_online_runners, n_agent_runners):
  worker_steps = n_runners * n_steps
  n_agent_runners = n_online_runners + n_agent_runners
  n_pbt_steps = ceil(worker_steps / n_agent_runners)
  assert n_agent_runners * n_pbt_steps >= worker_steps, (n_agent_runners, n_pbt_steps)

  return n_pbt_steps


class Controller(ControllerBase):
  @timeit
  def train(
    self, 
    agent_manager: AgentManager, 
    runner_manager: RunnerManager, 
    max_steps: int
  ):
    agent_manager.start_training()
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
      model_weights = self._retrieve_model_weights()
      steps = sum(runner_manager.run_with_model_weights(model_weights))
      self._steps += self.steps_per_run
      
      is_score_met = self._check_scores()
      if is_score_met:
        break

    status = Status.SCORE_MET if is_score_met else Status.TIMEOUT
    self._finish_iteration(eval_pids, status=status)

  """ Implementation for <pbt_train> """
  def _prepare_configs(self, n_runners: int, n_steps: int, iteration: int):
    configs = [dict2AttrDict(c, to_copy=True) for c in self.configs]
    runner_stats = ray.get(self.parameter_server.get_runner_stats.remote())
    assert self._iteration == runner_stats.iteration, (self._iteration, runner_stats.iteration)
    n_online_runners = runner_stats.n_online_runners
    n_agent_runners = runner_stats.n_agent_runners
    n_pbt_steps = _compute_pbt_steps(
      n_runners, 
      n_steps, 
      n_online_runners, 
      n_agent_runners, 
    )
    for c in configs:
      c.trainer.n_runners = n_online_runners + n_agent_runners
      c.buffer.n_runners = n_online_runners + n_agent_runners
      c.trainer.n_steps = n_pbt_steps
      c.runner.n_steps = n_pbt_steps
      c.buffer.n_steps = n_pbt_steps
    runner_stats.n_pbt_steps = n_pbt_steps
    do_logging(runner_stats, prefix=f'Runner Stats at Iteration {self._iteration}', color='blue')
    self._log_stats(runner_stats, self._iteration)
    return configs
  