import os
from math import ceil
from typing import Dict
import ray

from tools.log import do_logging
from core.typing import dict2AttrDict
from tools.file import write_file
from tools.timer import Every, timeit, Timer
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
    max_steps: int, 
    periods: Dict[str, Every]
  ):
    agent_manager.start_training()
    eval_pids = []

    while self._steps < max_steps:
      eval_pids = self._pretrain(periods, eval_pids)

      model_weights = self._retrieve_model_weights()
      # runner_manager.run_with_model_weights(model_weights)
      # self._steps += self.steps_per_run
      try:
        with Timer('run', 1):
          runner_manager.run_with_model_weights(model_weights)
        self._steps += self.steps_per_run
      except Exception as e:
        do_logging(e)
        write_file(os.path.join(self.dir, 'error.txt'), str(e))
        runner_manager.destroy_runners()
        runner_manager.build_runners(
          self.configs, 
          remote_buffers=self.agent_manager.get_agents(), 
          active_models=self.active_models
        )
        agent_manager.clear_buffer()
      # In order to use the following line 
      # first make sure buffer is succeessfully merged
      # agent_manager.train()

      self._posttrain(periods, eval_pids)
      if self._check_termination(self._steps, max_steps):
        break

    self._finish_iteration(eval_pids)

  """ Implementation for <pbt_train> """
  def _prepare_configs(self, n_runners: int, n_steps: int, iteration: int):
    configs = [dict2AttrDict(c, to_copy=True) for c in self.configs]
    runner_stats = ray.get(self.parameter_server.get_runner_stats.remote()) # 获取runner的信息
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
    self.pbt_monitor.store(**runner_stats)
    self.pbt_monitor.record(self._iteration, print_terminal_info=True)
    return configs
  