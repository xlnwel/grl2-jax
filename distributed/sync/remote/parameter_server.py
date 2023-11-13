from distributed.common.remote.parameter_server import ParameterServer as ParameterServerBase


class ParameterServer(ParameterServerBase):
  def _reset_prepared_strategy(self, rid=None):
    if rid < 0:
      self._prepared_strategies = [
        [None for _ in range(self.n_agents)] 
        for _ in range(self.n_runners)
      ]
      self._ready = [False] * self.n_runners
    else:
      self._prepared_strategies[rid] = [None for _ in range(self.n_agents)]
      self._ready[rid] = False
