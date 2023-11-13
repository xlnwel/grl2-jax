import time
import ray

from distributed.common.local.agent_manager import AgentManager
from distributed.common.local.runner_manager import RunnerManager
from distributed.common.local.controller import Controller as ControllerBase
from tools.timer import Every, timeit


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
    to_eval = Every(self.config.eval_period, final=max_steps)
    to_store = Every(self.config.store_period, final=max_steps)
    eval_pids = []
    while self._steps < max_steps:
      self._preprocessing(to_eval, to_restart_runners, to_store, eval_pids)
      time.sleep(1)

      steps = runner_manager.get_total_steps()
      self._steps = sum(steps)
      # do_logging(f'finishing sampling: total steps={steps}')

    self._finish_iteration(eval_pids)
