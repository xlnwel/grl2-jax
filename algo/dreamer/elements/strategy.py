import functools
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory as MemoryBase
from core.typing import AttrDict
from jax_tools import jax_utils


class Memory(MemoryBase):
    def __init__(self, model):
        """ Setup attributes for RNNS """
        self.model = model
        self._state = None
        self._state_rssm = None
        self._obs_rssm = None

    def add_memory_state_to_input(self, 
            inp: dict, reset: np.ndarray, state: NamedTuple=None,
            state_rssm: NamedTuple=None, obs_rssm: NamedTuple=None):
        """ Adds memory state and mask to the input. """
        # Check consistency
        assert (state is None and state_rssm is None and obs_rssm is None) or \
            (state is not None and state_rssm is not None and obs_rssm is not None)

        if state is None and self._state is None:
            self._state, self._state_rssm, self._obs_rssm = \
                self.model.get_initial_state(next(iter(inp.values())).shape[0])

        if state is None:
            state = self._state
            state_rssm = self._state_rssm
            obs_rssm = self._obs_rssm

        state, state_rssm, obs_rssm = self.apply_reset_to_state(state, state_rssm, obs_rssm, reset)
        inp.update({
            'state_reset': reset, 
            'state': state,
            'state_rssm': state_rssm,
            'obs_rssm': obs_rssm,
        })

        return inp

    def apply_reset_to_state(self, state: NamedTuple, state_rssm: NamedTuple, obs_rssm: NamedTuple, reset: np.ndarray):
        # state.shape: [B, U, D]; reset.shape: [B, U]
        assert state is not None, state
        reset = jnp.expand_dims(reset, -1)
        state = jax_utils.tree_map(lambda x: x*(1-reset), state)
        state_rssm = jax_utils.tree_map(lambda x: x*(1-reset), state_rssm)
        obs_rssm = jax_utils.tree_map(lambda x: x*(1-reset), obs_rssm)
        return state, state_rssm, obs_rssm

    def reset_states(self):
        self._state = None
        self._state_rssm = None
        self._obs_rssm = None

    def set_states(self, state: NamedTuple=None, state_rssm: NamedTuple=None, obs_rssm: NamedTuple=None):
        self._state = state
        self._state_rssm = state_rssm
        self._obs_rssm = obs_rssm

    def get_states(self):
        return self._state, self._state_rssm, self._obs_rssm

    def get_states_for_inputs(self, **kwargs):
        if self._state is None:
            self._state, self._state_rssm, self._obs_rssm = self.model.get_initial_state(**kwargs)
        return self._state, self._state_rssm, self._obs_rssm


class Strategy(StrategyBase):
    def _post_init(self):
        if self.actor is not None:
            self._memory = Memory(self.model)

    def _prepare_input_to_actor(self, env_output):
        """ Extract data from env_output as the input 
        to Actor for inference """
        if isinstance(env_output.obs, list):
            assert len(env_output.obs) == 1, env_output.obs
            inp = env_output.obs[0]
        else:
            inp = env_output.obs
        inp.update({
            "prev_action": env_output.prev_action
        })
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)
        return inp

    def _record_output(self, out):
        state, state_post, obs_post = out[-3:]
        self._memory.set_states(state, state_post, obs_post)
    
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

create_strategy = functools.partial(create_strategy, strategy_cls=Strategy)
