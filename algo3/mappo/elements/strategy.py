import functools
import logging
from typing import Dict
import numpy as np

from .trainloop import PPOTrainingLoop
from core.elements.strategy import Strategy, create_strategy
from core.mixin.strategy import Memory
from utility.utils import concat_map


logger = logging.getLogger(__name__)


class MAPPOStrategy(Strategy):
    """ Initialization """
    def _post_init(self):
        self._value_inp = None

        if self.actor is not None:
            self._memory = Memory(self.model)
        if self.trainer is not None:
            env_stats = self.trainer.env_stats

            state_keys = self.model.state_keys
            mid = len(state_keys) // 2
            value_state_keys = state_keys[mid:]
            self._value_sample_keys = [
                'global_state', 'value', 'traj_ret', 'mask'
            ] + list(value_state_keys)
            if env_stats.use_life_mask:
                self._value_sample_keys.append('life_mask')
            self._n_players = env_stats.n_players

    """ Calling Methods """
    def _prepare_input_to_actor(self, env_output):
        inp = env_output.obs
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)

        return inp

    def _record_output(self, out):
        state = out[-1]
        self._memory.set_states(state)

    """ PPO Methods """
    def record_inputs_to_vf(self, env_output):
        value_input = concat_map({'global_state': env_output.obs['global_state']})
        value_input = self.actor.process_obs_with_rms(
            value_input, update_rms=False)
        reset = concat_map(env_output.reset)
        state = self._memory.get_states()
        mid = len(state) // 2
        state = self.model.value_state_type(*state[mid:])
        self._value_inp = self._memory.add_memory_state_to_input(
            value_input, reset, state=state)

    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure global_state is normalized if obs normalization is required
        if value_inp is None:
            value_inp = self._value_inp
        value, _ = self.model.compute_value(**value_inp)
        value = value.numpy().reshape(-1, self._n_players)
        return value


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=MAPPOStrategy,
    training_loop_cls=PPOTrainingLoop,
)
