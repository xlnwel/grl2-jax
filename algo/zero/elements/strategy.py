import functools
import logging
from typing import Dict
import numpy as np

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory

logger = logging.getLogger(__name__)


class Strategy(StrategyBase):
    def _post_init(self):
        self._value_inp = {}
        if self.actor is not None and self.model.state_type:
            self._memory = Memory(self.model)

    def record_inputs_to_vf(self, env_output):
        self._value_inp['obs'] = self.actor.normalize_obs(
            env_output.obs)

    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure you normalize obs first if obs normalization is required
        if value_inp is None:
            value_inp = self._value_inp
        if 'global_state' in value_inp:
            value_inp['obs'] = value_inp.pop('global_state')
        value, _ = self.model.compute_value(**value_inp)
        return value.numpy()


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=Strategy,
)
