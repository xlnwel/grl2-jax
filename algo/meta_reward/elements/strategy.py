import functools

from core.elements.strategy import Strategy as StrategyBase, create_strategy


class Strategy(StrategyBase):
    def _post_init(self):
        self._prev_sid = None
        self._prev_idx = None
        self._prev_event = None
        self._prev_global_state = None
    
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
        if 'global_state'  in inp and self._prev_global_state is None:
            self._prev_global_state = inp['global_state'].copy()
        inp['prev_sid'] = self._prev_sid
        inp['prev_idx'] = self._prev_idx
        inp['prev_event'] = self._prev_event
        inp['prev_global_state'] = self._prev_global_state
        return inp

    def _record_output(self, out):
        super()._record_output(out)
        _, stats, _ = out
        if 'sid' in stats:
            self._prev_sid = stats.pop('sid')
        if 'idx' in stats:
            self._prev_idx = stats.pop('idx')
        if 'event' in stats:
            self._prev_event = stats.pop('event')
        if 'global_state' in stats:
            self._prev_global_state = stats.pop('global_state')


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=Strategy,
)

