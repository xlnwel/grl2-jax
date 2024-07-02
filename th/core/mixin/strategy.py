import numpy as np
import jax

from typing import NamedTuple
from numpy import ndarray
from core.mixin.strategy import Memory as MemoryBase
from tools.utils import tree_map


class Memory(MemoryBase):
  def apply_reset_to_state(self, state: NamedTuple, reset: ndarray):
    if state is None:
      return
    reset = np.expand_dims(reset, (-2, -1))
    state = torch.utils._pytree.tree_map(lambda x: x*(1-reset), state)
    return state
