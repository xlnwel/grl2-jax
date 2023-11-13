import time
import ray

from distributed.common.remote.runner import MultiAgentRunner as MultiAgentRunnerBase


class MultiAgentRunner(MultiAgentRunnerBase):

  def run_loop(self):
    # wait for asynchronous sampling
    time.sleep(self.id)
    while self.run_signal:
      mids = ray.get(self.parameter_server.get_strategies.remote(self.id))
      self.run_with_model_weights(mids)
