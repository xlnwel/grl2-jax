import functools

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory
from core.typing import AttrDict


class Strategy(StrategyBase):
    def _post_init(self):
        if self.actor is not None:
            self._memory = Memory(self.model)

    def _prepare_input_to_actor(self, env_output):
        inp = super()._prepare_input_to_actor(env_output)
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)
        return inp

    def _record_output(self, out):
        state = out[-1]
        self._memory.set_states(state)

    def compute_value(self, env_output):
        inp = AttrDict(global_state=env_output.obs['global_state'])
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)
        value = self.model.compute_value(inp)
        return value
    
    def reset_states(self):
        self._memory.reset_states()

    def set_states(self, state):
        self._memory.set_states(state)
    
    def get_states(self):
        state = self._memory.get_states()
        self._memory.reset_states()
        return state


create_strategy = functools.partial(create_strategy, strategy_cls=Strategy)
