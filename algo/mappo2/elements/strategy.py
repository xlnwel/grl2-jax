import functools
from typing import Dict
import numpy as np

from core.elements.strategy import create_strategy
from algo.ppo.elements.strategy import PPOTrainingLoop
from algo.mappo.elements.strategy import MAPPOStrategy


class MAPPO2Strategy(MAPPOStrategy):
    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure global_state is normalized if obs normalization is required
        if value_inp is None:
            value_inp = self._value_input
        value, _ = self.model.compute_value(**value_inp)
        value = value.numpy()
        return value


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=MAPPO2Strategy,
    training_loop_cls=PPOTrainingLoop
)
