import functools
import logging

from .trainloop import PPOTrainingLoop
from core.elements.strategy import Strategy, create_strategy
from core.mixin.strategy import Memory
from env.typing import EnvOutput

logger = logging.getLogger(__name__)


class MAPPOStrategy(Strategy):
    """ Initialization """
    def _post_init(self):
        if self.actor is not None and self.model.state_type:
            self._memory = Memory(self.model)

    """ Calling Methods """
    def _prepare_input_to_actor(self, env_output: EnvOutput):
        inp = super()._prepare_input_to_actor(env_output)
        if self.model.state_type:
            state = self._memory.get_states_for_inputs(batch_size=env_output.reward.size)
            inp = self._memory.add_memory_state_to_input(inp, env_output.reset, state=state)
        return inp

    def _record_output(self, out):
        if self.model.state_type:
            state = out[-1]
            self._memory.set_states(state)


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=MAPPOStrategy,
    training_loop_cls=PPOTrainingLoop,
)
