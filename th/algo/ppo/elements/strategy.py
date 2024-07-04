from functools import partial

from core.typing import AttrDict
from th.elements.strategy import Strategy as StrategyBase, create_strategy
from tools.run import concat_along_unit_dim
from tools.utils import batch_dicts


class Strategy(StrategyBase):
  def compute_value(self, env_output, states=None):
    if isinstance(env_output.obs, list):
      inp = batch_dicts([AttrDict(
        global_state=o.get('global_state', o['obs'])
      ) for o in env_output.obs], concat_along_unit_dim)
      reset = concat_along_unit_dim(env_output.reset)
    else:
      inp = AttrDict(
        global_state=env_output.obs.get('global_state', env_output.obs['obs'])
      )
      reset = env_output.reset
    inp = self._memory.add_memory_state_to_input(inp, reset, states)
    value = self.actor.compute_value(inp)

    return value


create_strategy = partial(create_strategy, strategy_cls=Strategy)
