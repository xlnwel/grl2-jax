from core.elements.actor import Actor as ActorBase
from typing import Tuple, Dict
import jax.numpy as jnp

from core.elements.model import Model
from core.mixin.actor import RMS
from core.typing import ModelPath, AttrDict, dict2AttrDict
from tools.utils import set_path

class Actor(ActorBase):
    def _process_output(
        self, 
        inp: dict, 
        out: Tuple[Dict[str, jnp.DeviceArray]], 
        evaluation: bool
    ):
        """ Post-processes output.
        
        Args:
            inp: Pre-processed inputs
            out: Model output
        Returns:
            (action, stats, rnn_state)
        """
        action, stats, state, state_post, obs_post = out
        if state is not None and not evaluation:
            prev_state = inp.state
            stats.update({
                'state_reset': inp['state_reset'], 
                'state': prev_state,
            })
        if self.config.get('update_obs_at_execution', True) \
            and not evaluation and self.rms is not None \
                and self.rms.is_obs_normalized:
            stats.update({k: inp[k] 
                for k in self.config.rms.obs_names})
        return action, stats, state, state_post, obs_post

def create_actor(config, model, name='model'):
    return Actor(config=config, model=model, name=name)
