from functools import partial

from core.typing import AttrDict
from algo.lka_common.elements.strategy import Strategy as StrategyBase, create_strategy


class Strategy(StrategyBase):
    def compute_value(self, env_output, states=None):
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
        inps = self._memory.add_memory_state_to_input(inps, resets, states)
        value = self.actor.compute_value(inps)
        return value


create_strategy = partial(create_strategy, strategy_cls=Strategy)
