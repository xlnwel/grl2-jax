import numpy as np

from core.elements.agent import create_agent


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