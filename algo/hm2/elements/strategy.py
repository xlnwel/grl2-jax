import functools
import logging

from .trainloop import MAPPOTrainingLoop
from core.elements.strategy import Strategy, create_strategy
from core.mixin.strategy import Memory

logger = logging.getLogger(__name__)


class MAPPOStrategy(Strategy):
    """ Initialization """
    def _post_init(self):
        self._value_inp = None

        if self.actor is not None and self.model.has_rnn:
            self._memory = Memory(self.model)

    """ Calling Methods """
    def _prepare_input_to_actor(self, env_output):
        inp = env_output.obs.copy()
        if self.model.has_rnn:
            state = self._memory.get_states_for_inputs(batch_size=env_output.reward.size)
            inp = self._memory.add_memory_state_to_input(inp, env_output.reset, state=state)
        return inp

    def _record_output(self, out):
        if self.model.has_rnn:
            state = out[-1]
            self._memory.set_states(state)


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=MAPPOStrategy,
    training_loop_cls=MAPPOTrainingLoop,
)
