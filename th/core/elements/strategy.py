import functools

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from th.core.mixin.strategy import Memory


class Strategy(StrategyBase):
  def _setup_memory_cls(self):
    self.memory_cls = Memory


create_strategy = functools.partial(create_strategy, strategy_cls=Strategy)
