from core.elements.actor import Actor as ActorBase
from typing import Tuple, Dict
import jax.numpy as jnp

from core.elements.model import Model
from core.mixin.actor import RMS
from core.typing import ModelPath, AttrDict, dict2AttrDict
from tools.utils import set_path

class Actor(ActorBase):
    def __call__(self, inp, evaluation):
        out = self.model.action(inp, evaluation)
        out = self._process_output(inp, out, evaluation)
        return out

    def _process_output(self, inp, out, evaluation):
        return out

    def model_rollout(self, state, rollout_length):
        rollout_data = self.model.model_rollout(state, rollout_length)
        return rollout_data

def create_actor(config, model, name='model'):
    return Actor(config=config, model=model, name=name)
