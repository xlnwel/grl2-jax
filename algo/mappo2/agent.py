import numpy as np
from algo.mappo.agent import create_agent, get_data_format

def collect(buffer, env, env_step, reset, reward, 
            discount, next_obs, **kwargs):
    kwargs['reward'] = np.concatenate(reward)
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = np.concatenate(discount)
    buffer.add(**kwargs)

def random_actor(env_output, env=None, **kwargs):
    obs = env_output.obs
    a = np.concatenate(env.random_action())
    terms = {
        'obs': np.concatenate(obs['obs']), 
        'global_state': np.concatenate(obs['global_state']),
    }
    return a, terms
