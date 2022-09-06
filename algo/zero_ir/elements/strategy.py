import numpy as np
import functools

from core.elements.strategy import Strategy as StrategyBase, create_strategy


class Strategy(StrategyBase):
    def _post_init(self):
        self._prev_sid = None
        self._prev_idx = None
        self._prev_event = None
        self._prev_hidden_state = None
    
    def _prepare_input_to_actor(self, env_output):
        inp = super()._prepare_input_to_actor(env_output)
        if isinstance(inp, list):
            assert len(inp) == 1, inp
            inp = inp[0]
        if 'sid' in inp and self._prev_sid is None:
            self._prev_sid = inp['sid'].copy()
        if 'idx' in inp and self._prev_idx is None:
            self._prev_idx = inp['idx'].copy()
        if 'event' in inp and self._prev_event is None:
            self._prev_event = inp['event'].copy()
        if 'hidden_state'  in inp and self._prev_hidden_state is None:
            self._prev_hidden_state = inp['hidden_state'].copy()
        inp['prev_sid'] = self._prev_sid
        inp['prev_idx'] = self._prev_idx
        inp['prev_event'] = self._prev_event
        inp['prev_hidden_state'] = self._prev_hidden_state
        return inp

    def _record_output(self, out):
        super()._record_output(out)
        _, terms, _ = out
        if 'sid' in terms:
            self._prev_sid = terms.pop('sid')
        if 'idx' in terms:
            self._prev_idx = terms.pop('idx')
        if 'event' in terms:
            self._prev_event = terms.pop('event')
        if 'hidden_state' in terms:
            self._prev_hidden_state = terms.pop('hidden_state')


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=Strategy,
)

