from functools import partial

from jx.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory
from tools.run import concat_along_unit_dim


class Strategy(StrategyBase):
  def _prepare_input_to_actor(self, env_output):
    inp = super()._prepare_input_to_actor(env_output)
    if isinstance(env_output.reset, list):
      reset = concat_along_unit_dim(env_output.reset)
    else:
      reset = env_output.reset
    inp = self._memory.add_memory_state_to_input(inp, reset)
    return inp

  def _record_output(self, out):
    state = out[-1]
    self._memory.set_states(state)
    return out

create_strategy = partial(create_strategy, strategy_cls=Strategy)
