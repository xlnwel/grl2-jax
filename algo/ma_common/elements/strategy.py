from functools import partial
from typing import List
import jax

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory as MemoryBase
from core.typing import AttrDict, dict2AttrDict
from tools.utils import batch_dicts
from algo.ma_common.run import concat_along_unit_dim


class Memory(MemoryBase):
    def add_memory_state_to_input(self, 
            inps: List, resets: List, states: List=None):
        if states is None and self._state is None:
            self._state = self.model.get_initial_state(
                next(iter(inps[0].values())).shape[0])

        if states is None:
            states = self._state
        if states is None:
            return inps  # no memory is maintained

        states = self.apply_reset_to_state(states, resets)
        for inp, state, reset in zip(inps, states, resets):
            inp.state = state
            inp.state_reset = reset
        
        return inps

    def apply_reset_to_state(self, states: List[AttrDict], resets: List):
        if states is None:
            return
        for state, reset in zip(states, resets):
            state = jax.tree_util.tree_map(lambda x: x*(1-reset), state)
        return state


class Strategy(StrategyBase):
    def _setup_memory_cls(self):
        self.memory_cls = Memory

    def _post_init(self):
        self.aid2uids = self.env_stats.aid2uids

    def _prepare_input_to_actor(self, env_output):
        if isinstance(env_output.obs, list):
            inps = [dict2AttrDict(o, to_copy=True) for o in env_output.obs]
            resets = env_output.reset
        else:
            inps = [env_output.obs.slice((slice(None), uids)) for uids in self.aid2uids]
            resets = [env_output.reset[:, uids] for uids in self.aid2uids]
        inps = self._memory.add_memory_state_to_input(inps, resets)

        return inps

    def _record_output(self, out):
        act, stats, states = out
        self._memory.set_states(states)
        if states is not None:
            states = batch_dicts(states, concat_along_unit_dim)
        return act, stats, states


create_strategy = partial(create_strategy, strategy_cls=Strategy)
