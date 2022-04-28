import functools
import logging
from typing import Dict
import numpy as np

from core.elements.strategy import Strategy, create_strategy
from algo.ppo.elements.trainloop import PPOTrainingLoop


logger = logging.getLogger(__name__)


class PPOStrategy(Strategy):
    def _post_init(self):
        self._value_inp = {}

    def record_inputs_to_vf(self, env_output):
        self._value_inp['global_state'] = self.actor.normalize_obs(env_output.obs)

    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure you normalize obs first if obs normalization is required
        if value_inp is None:
            value_inp = self._value_inp
        value, _ = self.model.compute_value(**value_inp)
        return value.numpy()


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=PPOStrategy,
    training_loop_cls=PPOTrainingLoop
)
