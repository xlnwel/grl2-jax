import time
from core.log import do_logging
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
      # agent_manager.train()

    self._finish_iteration(eval_pids)
