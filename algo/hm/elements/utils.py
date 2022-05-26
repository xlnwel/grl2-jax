import numpy as np
import tensorflow as tf

from utility.typing import AttrDict


def update_data_format_with_rnn_states(
    data_format, 
    config, 
    basic_shape, 
    model, 
):
    if config.get('store_state'):
        if config.get('actor_rnn_type') or config.get('value_rnn_type'):
            data_format['mask'] = (basic_shape, tf.float32, 'mask')
            dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        if config.get('actor_rnn_type'):
            data_format['actor_state'] = model.policy.state_type(
                *[((None, sz), dtype, name) 
                for name, sz in model.policy.state_size._asdict().items()])
        if config.get('value_rnn_type'):
            data_format['value_state'] = model.value.state_type(
                *[((None, sz), dtype, name) 
                for name, sz in model.value.state_size._asdict().items()])

    return data_format

def get_basics(
    config: AttrDict, 
    env_stats: AttrDict, 
    model
):
    if 'aid' in config:
        aid = config.aid
        n_units = len(env_stats.aid2uids[aid])
        basic_shape = (None, config['sample_size'], n_units) \
            if model.has_rnn else (None, n_units)
        shapes = env_stats['obs_shape'][aid]
        dtypes = env_stats['obs_dtype'][aid]

        action_shape = env_stats.action_shape[aid]
        action_dim = env_stats.action_dim[aid]
        action_dtype = env_stats.action_dtype[aid]
    else:
        basic_shape = (None, config['sample_size']) \
            if model.has_rnn else (None,)
        shapes = env_stats['obs_shape']
        dtypes = env_stats['obs_dtype']

        action_shape = env_stats.action_shape
        action_dim = env_stats.action_dim
        action_dtype = env_stats.action_dtype

    return basic_shape, shapes, dtypes, action_shape, action_dim, action_dtype

def get_data_format(
    config: AttrDict, 
    env_stats: AttrDict, 
    model
):
    basic_shape, shapes, dtypes, action_shape, \
        action_dim, action_dtype = \
        get_basics(config, env_stats, model)
    
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}

    data_format.update(dict(
        action=((*basic_shape, *action_shape), action_dtype, 'action'),
        reward=(basic_shape, tf.float32, 'reward'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logprob=(basic_shape, tf.float32, 'logprob'),
    ))

    data_format = update_data_format_with_rnn_states(
        data_format,
        config,
        basic_shape,
        model
    )

    return data_format

def collect(buffer, env, env_step, reset, next_obs_dict, **kwargs):
    next_obs = np.where(np.expand_dims(reset, -1), env.prev_obs(), next_obs_dict['obs'])
    buffer.add(**kwargs, reset=reset, next_obs=next_obs)
