from distributed.common.remote.parameter_server_sp import \
  SPParameterServer as SPParameterServerBase, ExploiterSPParameterServer as ExploiterSPParameterServerBase


class SPParameterServer(SPParameterServerBase):
  def _reset_prepared_strategy(self, rid=None):
    pass


class ExploiterSPParameterServer(ExploiterSPParameterServerBase):
  def _reset_prepared_strategy(self, rid=None):
    pass
