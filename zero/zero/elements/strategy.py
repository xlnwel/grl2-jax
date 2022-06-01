import functools
import logging
from typing import Dict
import numpy as np

from .trainloop import PPOTrainingLoop
from core.elements.strategy import Strategy, create_strategy
from core.mixin.strategy import Memory
from env.typing import EnvOutput

logger = logging.getLogger(__name__)


class PPOStrategy(Strategy):
    """ Initialization """
    def _post_init(self):
        self._value_inp = {}
        if self.actor is not None and self.model.state_type:
            self._memory = Memory(self.model)

    """ Calling Methods """
    def _prepare_input_to_actor(self, env_output: EnvOutput):
        inp = super()._prepare_input_to_actor(env_output)
        if self.model.state_type:
            state = self._memory.get_states_for_inputs(
                batch_size=env_output.reward.size)
            inp = self._memory.add_memory_state_to_input(
                inp, env_output.reset, state=state)
        return inp

    def _record_output(self, out):
        if self.model.state_type:
            state = out[-1]
            self._memory.set_states(state)

    def record_inputs_to_vf(self, env_output):
        name = 'global_state' if 'global_state' in env_output.obs \
            else 'obs'
        self._value_inp['global_state'] = self.actor.normalize_obs(
            env_output.obs, name=name)

    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure you normalize obs first if obs normalization is required
        if value_inp is None:
            value_inp = self._value_inp
        value, _ = self.model.compute_value(**value_inp)
        return value.numpy()



create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=PPOStrategy,
    training_loop_cls=PPOTrainingLoop,
)
