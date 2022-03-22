import tensorflow as tf


def get_data_format(config, env_stats, model):
    aid = config.aid
    n_units = len(env_stats.aid2uids[aid])
    basic_shape = (None, config['sample_size'], n_units) \
        if config.get('rnn_type') and config.get('sample_size') else (None, n_units)
    shapes = env_stats['obs_shape'][aid]
    dtypes = env_stats['obs_dtype'][aid]
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}

    action_shape = env_stats.action_shape[aid]
    action_dtype = env_stats.action_dtype[aid]
    data_format.update(dict(
        action=((*basic_shape, *action_shape), action_dtype, 'action'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logpi=(basic_shape, tf.float32, 'logpi'),
    ))

    if config.get('rnn_type') and config.get('store_state'):
        data_format['mask'] = (basic_shape, tf.float32, 'mask')
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        state_type = type(model.state_size)
        data_format['state'] = state_type(*[((None, sz), dtype, name) 
            for name, sz in model.state_size._asdict().items()])

    return data_format
