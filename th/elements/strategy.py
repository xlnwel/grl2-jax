import numpy as np

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory as MemoryBase
from tools.tree_ops import tree_map


class Memory(MemoryBase):
  def apply_reset_to_state(self, state, reset):
    if state is None:
      return
    reset = np.expand_dims(reset, (-2, -1))
    state = tree_map(lambda x: x*(1-reset), state)
    return state


class Strategy(StrategyBase):
  def _setup_memory_cls(self):
    self.memory_cls = Memory
