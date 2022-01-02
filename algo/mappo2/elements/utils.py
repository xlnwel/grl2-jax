import numpy as np

from algo.mappo.elements.utils import *


def collect(buffer, env, env_step, reset, reward, discount,
            next_obs, **kwargs):
    kwargs['reward'] = np.concatenate(reward)
    kwargs['discount'] = np.concatenate(discount)

    buffer.add(**kwargs)

def random_actor(env_output, env=None, **kwargs):
    obs = env_output.obs
    a = np.concatenate(env.random_action())
    terms = {
        'obs': np.concatenate(obs['obs']), 
        'global_state': np.concatenate(obs['global_state']),
        'life_mask': np.concatenate(obs['life_mask']),
    }
    return a, terms
