import functools
import jax
import jax.numpy as jnp

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
        inp = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), inp)
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)
        if isinstance(inp.state, dict) and 'value' in inp.state:
            inp['state'] = inp.state['value']
        value = self.model.compute_value(inp)
        value = jnp.squeeze(value, 1)
        return value

    def lookahead_train(self, **kwargs):
        return self.train_loop.lookahead_train(**kwargs)


create_strategy = functools.partial(create_strategy, strategy_cls=Strategy)
