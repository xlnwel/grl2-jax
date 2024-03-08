from distributed.common.remote.parameter_server import ParameterServer as ParameterServerBase


class ParameterServer(ParameterServerBase):
  def _reset_prepared_strategy(self, rid=-1):
    pass
