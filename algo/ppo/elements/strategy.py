from functools import partial

from core.typing import dict2AttrDict
from core.elements.strategy import Strategy as StrategyBase, create_strategy
from tools.run import concat_along_unit_dim
from tools.utils import batch_dicts


class Strategy(StrategyBase):
  def compute_value(self, env_output, states=None):
    if isinstance(env_output.obs, list):
      inp = batch_dicts(env_output.obs, concat_along_unit_dim)
      reset = concat_along_unit_dim(env_output.reset)
    else:
      inp = dict2AttrDict(env_output.obs)
      inp.setdefault('global_state', inp['obs'])
      reset = env_output.reset
    inp = self._memory.add_memory_state_to_input(inp, reset, states)
    value = self.actor.compute_value(inp)

    return value


create_strategy = partial(create_strategy, strategy_cls=Strategy)
