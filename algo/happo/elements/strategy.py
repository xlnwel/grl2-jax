from functools import partial

from core.typing import AttrDict
from algo.lka_common.elements.strategy import Strategy as StrategyBase, create_strategy


class Strategy(StrategyBase):
    def _prepare_input_to_actor(self, env_output):
        inp = super()._prepare_input_to_actor(env_output)
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)
        return inp

    def _record_output(self, out):
        state = out[-1]
        self._memory.set_states(state)
        return out

    def compute_value(self, env_output, states=None):
        if isinstance(env_output.obs, list):
            inps = [AttrDict(
                global_state=o.get('global_state', o['obs'])
            ) for o in env_output.obs]
            resets = env_output.reset
        else:
            inps = [AttrDict(
                global_state=env_output.obs.get('global_state', env_output.obs['obs'])[:, uids]
            ) for uids in self.aid2uids]
            resets = [env_output.reset[:, uids] for uids in self.aid2uids]
        inps = self._memory.add_memory_state_to_input(inps, resets, states)
        value = self.actor.compute_value(inps)

        return value


create_strategy = partial(create_strategy, strategy_cls=Strategy)
