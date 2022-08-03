from types import FunctionType
from typing import Tuple
import numpy as np
import tensorflow as tf

from utility.typing import AttrDict


def compute_inner_steps(config):
    if config.K is not None and config.L is not None:
        config.inner_steps = config.K + config.L
    else:
        config.inner_steps = None
    return config

def get_basics(
    config: AttrDict, 
    env_stats: AttrDict, 
    model
):
    if 'aid' in config:
        aid = config.aid
        n_units = len(env_stats.aid2uids[aid])
        basic_shape = (None, config['sample_size'], n_units)
        shapes = env_stats['obs_shape'][aid]
        dtypes = env_stats['obs_dtype'][aid]

        action_shape = env_stats.action_shape[aid]
        action_dim = env_stats.action_dim[aid]
        action_dtype = env_stats.action_dtype[aid]
    else:
        basic_shape = (None, config['sample_size'], 1)
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
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        state_type = type(model.state_size)
        data_format['mask'] = (basic_shape, tf.float32, 'mask')
        data_format['state'] = state_type(*[((None, sz), dtype, name) 
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
        get_basics(config, env_stats, model)
    if meta:
        basic_shape = (config.inner_steps + config.extra_meta_step,) + basic_shape
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
        value=(basic_shape, tf.float32, 'value'),
        reward=(basic_shape, tf.float32, 'reward'),
        discount=(basic_shape, tf.float32, 'discount'),
        reset=(basic_shape, tf.float32, 'reset'),
        mu_logprob=(basic_shape, tf.float32, 'mu_logprob'),
    ))
    if env_stats.is_action_discrete:
        data_format['mu'] = ((*basic_shape, action_dim), tf.float32, 'mu')
    else:
        data_format['mu_mean'] = ((*basic_shape, action_dim), tf.float32, 'mu_mean')
        data_format['mu_std'] = ((*basic_shape, action_dim), tf.float32, 'mu_std')
    
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
    if buffer.config.timeout_done:
        next_obs = next_obs['obs']
    else:
        next_obs = np.where(
            np.expand_dims(reset, -1), 
            env.prev_obs()['obs'], 
            next_obs['obs']
        )

    buffer.add(**kwargs, reset=reset, next_obs=next_obs)
