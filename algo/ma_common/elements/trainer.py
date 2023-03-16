import numpy as np
import jax.numpy as jnp

from core.typing import dict2AttrDict
from tools.display import print_dict_info


def construct_fake_data(env_stats, aid):
    b = 8
    s = 400
    u = len(env_stats.aid2uids[aid])
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    basic_shape = (b, s, u)
    data = {k: jnp.zeros((b, s+1, u, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.value = jnp.zeros(basic_shape, jnp.float32)
    data.reward = jnp.zeros(basic_shape, jnp.float32)
    data.discount = jnp.zeros(basic_shape, jnp.float32)
    data.reset = jnp.zeros(basic_shape, jnp.float32)
    data.mu_logprob = jnp.zeros(basic_shape, jnp.float32)
    data.mu_logits = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.advantage = jnp.zeros(basic_shape, jnp.float32)
    data.v_target = jnp.zeros(basic_shape, jnp.float32)

    print_dict_info(data)
    
    return data


def sample_stats(stats, max_record_size=10):
    # we only sample a small amount of data to reduce the cost
    stats = dict2AttrDict({k if '/' in k else f'train/{k}': 
        np.random.choice(stats[k].reshape(-1), max_record_size) 
        if isinstance(stats[k], (np.ndarray, jnp.DeviceArray)) \
            and stats[k].size > max_record_size else stats[k] 
        for k in sorted(stats.keys())})
    return stats
