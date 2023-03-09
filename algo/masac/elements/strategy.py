import functools
import jax.numpy as jnp

from core.elements.strategy import Strategy as StrategyBase, create_strategy
from core.mixin.strategy import Memory
from core.typing import AttrDict, dict2AttrDict


class Strategy(StrategyBase):
    def _post_init(self):
        self.aid2uids = self.env_stats.aid2uids
        if self.actor is not None:
            self._memory = Memory(self.model)

    def _prepare_input_to_actor(self, env_output):
        if isinstance(env_output.obs, list):
            inp = [dict2AttrDict(o) for o in env_output.obs]
        else:
            inp = [env_output.obs.slice((slice(None), uids)) for uids in self.aid2uids]
        return inp

    def _record_output(self, out):
        state = out[-1]

    def lookahead_train(self, **kwargs):
        return self.train_loop.lookahead_train(**kwargs)


create_strategy = functools.partial(create_strategy, strategy_cls=Strategy)
