from functools import partial
import jax
import jax.numpy as jnp

from core.typing import AttrDict
from algo.lka_common.elements.strategy import Strategy as StrategyBase, create_strategy


class Strategy(StrategyBase):
    def compute_value(self, env_output):
        if isinstance(env_output.obs, list):
            inps = [AttrDict(
                global_state=o['global_state']
            ) for o in env_output.obs]
            resets = env_output.reset
        else:
            inps = [AttrDict(
                global_state=env_output.obs['global_state'][:, uids]
            ) for uids in self.aid2uids]
            resets = [env_output.reset[:, uids] for uids in self.aid2uids]
        if self.model.has_rnn:
            # add the time dimension only when RNN is involved
            inps = jax.tree_util.tree_map(inps, lambda x: jnp.expand_dims(x, 1))
        inps = self._memory.add_memory_state_to_input(inps, resets)
        value = self.model.compute_value(inps)
        if self.model.has_rnn:
            # remove the time dimension
            value = jnp.squeeze(value, 1)
        return value


create_strategy = partial(create_strategy, strategy_cls=Strategy)
