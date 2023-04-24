import collections
from jax import nn
import jax.numpy as jnp

from core.typing import AttrDict, dict2AttrDict


DynamicsOutput = collections.namedtuple('dynamics', 'model reward discount')


def prefix_name(terms, name):
    if name is not None:
        new_terms = AttrDict()
        for k, v in terms.items():
            if '/' not in k:
                new_terms[f'{name}/{k}'] = v
            else:
                new_terms[k] = v
        return new_terms
    return terms


def bound_logvar(logvar, max_logvar, min_logvar):
    logvar = max_logvar - nn.softplus(max_logvar - logvar)
    logvar = min_logvar + nn.softplus(logvar - min_logvar)

    return logvar


def joint_actions(actions):
    all_actions = [actions]
    # roll along the unit dimension
    for _ in range(1, actions.shape[-2]):
        actions = jnp.roll(actions, 1, -2)
        all_actions.append(actions)
    all_actions = jnp.concatenate(all_actions, -1)

    return all_actions


def combine_sa(x, a):
    a = joint_actions(a)
    x = jnp.concatenate([x, a], -1)

    return x


def get_data_format(config, env_stats, model):
    aid = model.aid
    batch_size = config.batch_size
    seqlen = config.seqlen
    n_units = env_stats.n_units
    obs_shapes = env_stats.obs_shape[aid]
    obs_dtypes = env_stats.obs_dtype[aid]
    basic_shape = (batch_size, seqlen, n_units)
    data = {
        k: jnp.zeros((batch_size, seqlen, n_units, *v), obs_dtypes[k]) 
        for k, v in obs_shapes.items()}
    data.update({
        f'next_{k}': jnp.zeros((batch_size, seqlen, n_units, *v), obs_dtypes[k]) 
        for k, v in obs_shapes.items()})
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    action_dim = env_stats.action_dim[aid]
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.reset = jnp.zeros(basic_shape, jnp.float32)
    data.reward = jnp.zeros(basic_shape, jnp.float32)
