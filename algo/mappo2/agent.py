from typing import Dict
import numpy as np
from algo.mappo.agent import get_data_format, MAPPOAgent


def collect(buffer, env, env_step, reset, reward, 
            next_obs, **kwargs):
    kwargs['reward'] = np.concatenate(reward)
    buffer.add(**kwargs)

def random_actor(env_output, env=None, **kwargs):
    obs = env_output.obs
    discount = env_output.discount
    discount[np.any(discount, 1)] = 1
    a = np.concatenate(env.random_action())
    terms = {
        'obs': np.concatenate(obs['obs']), 
        'global_state': np.concatenate(obs['global_state']),
        'life_mask': np.concatenate(obs['life_mask']),
        'discount': np.concatenate(discount)
    }
    return a, terms

class MAPPO2Agent(MAPPOAgent):
    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure global_state is normalized if obs normalization is required
        if value_inp is None:
            value_inp = self._value_input
        value, _ = self.model.compute_value(**value_inp)
        value = value.numpy()
        return value

def create_agent(**kwargs):
    return MAPPO2Agent(**kwargs)
