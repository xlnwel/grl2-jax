import functools
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory as MemoryBase
from core.typing import AttrDict, dict2AttrDict
from jax_tools import jax_utils

class Memory(MemoryBase):

    def add_memory_state_to_input(self,
        inp: list, state: NamedTuple=None):
        if state is None and self._state is None:
            self._state = self.model.get_initial_state(
                next(iter(inp[0].values())).shape[0])
 
        if state is None:
            state = self._state

        for i in range(len(inp)):
            astate = jax_utils.tree_map(lambda x: x[..., i:i+1, :], state)
            astate = self.apply_reset_to_state(astate, inp[i].state_reset)
            inp[i].state = astate
        return inp

    def apply_reset_to_state(self, state: NamedTuple, reset: np.ndarray):
        assert state is not None, state
        if hasattr(state.policy, 'cell'):
            basic_shape = state.policy.cell.shape[:2]
        else:
            basic_shape = state.policy.shape[:2]
        reset = reset.reshape((*basic_shape, 1))
        state = jax_utils.tree_map(lambda x: x*(1-reset), state)
        return state

class Strategy(StrategyBase):
    def _post_init(self):
        if self.actor is not None:
            self._memory = Memory(self.model)

    def _prepare_input_to_actor(self, env_output):
        """ Extract data from env_output as the input 
        to Actor for inference """
        if isinstance(env_output, list):
            inp = [dict2AttrDict(o.obs) for o in env_output]
            for i in range(len(env_output)):
                inp[i].prev_action = env_output[i].prev_action
                inp[i].state_reset = env_output[i].reset
        else:
            inp = env_output.obs
            inp.update({
                'prev_action': env_output.prev_action
            })
            inp.state_reset = env_output.reset
        inp = self._memory.add_memory_state_to_input(inp)
        return inp
    
    def model_rollout(self, state, rollout_length):
        return self.actor.model_rollout(state, rollout_length)

    def compute_value(self, env_output):
        inp = AttrDict(global_state=env_output.obs['global_state'], action=env_output.prev_action)
        inp = jax_utils.tree_map(lambda x: jnp.expand_dims(x, 1), inp)
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)
        if isinstance(inp.state, dict) and 'value' in inp.state:
            inp['state'] = inp.state['value']
        value = self.model.compute_value(inp)
        value = jnp.squeeze(value, 1)
        return value

    def model_train_record(self, **kwargs):
        n, stats = self.train_loop.model_train(
            self.step_counter.get_train_step(), **kwargs)
        self.step_counter.add_train_step(n)
        return stats

    def train_record(self, data, **kwargs):
        n, stats = self.train_loop.train(
            self.step_counter.get_train_step(), data, **kwargs)
        self.step_counter.add_train_step(n)
        return stats


create_strategy = functools.partial(create_strategy, strategy_cls=Strategy)
