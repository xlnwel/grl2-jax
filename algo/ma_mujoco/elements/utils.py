from types import FunctionType
from typing import Tuple
import numpy as np
from jax import lax
import jax.numpy as jnp

from core.typing import AttrDict
from tools.utils import batch_dicts


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

def get_basics(
    config: AttrDict, 
    env_stats: AttrDict, 
):
    if 'aid' in config:
        aid = config.aid
        n_units = len(env_stats.aid2uids[aid])
        n_envs = config.n_runners * config.n_envs
        basic_shape = (n_envs, config['n_steps'], n_units)
        shapes = env_stats['obs_shape'][aid]
        dtypes = env_stats['obs_dtype'][aid]

        action_shape = env_stats.action_shape[aid]
        action_dim = env_stats.action_dim[aid]
        action_dtype = env_stats.action_dtype[aid]
    else:
        n_envs = config.n_runners * config.n_envs
        basic_shape = (n_envs, config['n_steps'], 1)
        shapes = env_stats['obs_shape']
        dtypes = env_stats['obs_dtype']

        action_shape = env_stats.action_shape
        action_dim = env_stats.action_dim
        action_dtype = env_stats.action_dtype

    return basic_shape, shapes, dtypes, action_shape, action_dim, action_dtype

def update_data_format_with_rnn_states(
    data_format: dict, 
    config: AttrDict, 
    basic_shape: Tuple, 
    model
):
    if config.get('store_state') and config.get('rnn_type'):
        assert model.state_size is not None, model.state_size
        state_type = type(model.state_size)
        data_format['mask'] = (basic_shape, np.float32, 'mask')
        data_format['state'] = state_type(*[((None, sz), np.float32, name) 
            for name, sz in model.state_size._asdict().items()])
    
    return data_format

def get_data_format(
    config: AttrDict, 
    env_stats: AttrDict, 
    model, 
    rnn_state_fn: FunctionType=update_data_format_with_rnn_states, 
    meta=False
):
    basic_shape, shapes, dtypes, action_shape, \
        action_dim, action_dtype = \
        get_basics(config, env_stats)
    if config.timeout_done:
        obs_shape = [s+1 if i == (2 if meta else 1) else s 
            for i, s in enumerate(basic_shape)]
        data_format = {k: ((*obs_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
    else:
        obs_shape = basic_shape
        data_format = {k: ((*obs_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
        data_format.update({f'next_{k}': ((*obs_shape, *v), dtypes[k], f'next_{k}') 
            for k, v in shapes.items()})

    data_format.update(dict(
        action=((*basic_shape, *action_shape), action_dtype, 'action'),
        value=(basic_shape, np.float32, 'value'),
        reward=(basic_shape, np.float32, 'reward'),
        discount=(basic_shape, np.float32, 'discount'),
        reset=(basic_shape, np.float32, 'reset'),
        mu_logprob=(basic_shape, np.float32, 'mu_logprob'),
    ))

    is_action_discrete = env_stats.is_action_discrete[config['aid']] \
        if isinstance(env_stats.is_action_discrete, list) else env_stats.is_action_discrete
    if is_action_discrete:
        data_format['mu'] = ((*basic_shape, action_dim), np.float32, 'mu')
    else:
        data_format['mu_mean'] = ((*basic_shape, action_dim), np.float32, 'mu_mean')
        data_format['mu_std'] = ((*basic_shape, action_dim), np.float32, 'mu_std')
    
    data_format = rnn_state_fn(
        data_format,
        config,
        basic_shape,
        model,
    )

    return data_format

def collect(buffer, env, env_step, reset, obs, next_obs, **kwargs):
    for k, v in obs.items():
        if k not in kwargs:
            kwargs[k] = v
    new_next_obs = batch_dicts(env.prev_obs(), func=lambda x: np.concatenate(x, -2))
    for k, v in new_next_obs.items():
        kwargs[f'next_{k}'] = v
    buffer.add(**kwargs, reset=reset)
    for i, r in enumerate(reset):
        if np.all(r):
            buffer.finish_episodes(i)
        for k in next_obs.keys():
            np.testing.assert_allclose(next_obs[k], new_next_obs[k])
