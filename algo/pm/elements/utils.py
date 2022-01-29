import tensorflow as tf


def get_data_format(config, env_stats, model):
    n_units = len(env_stats.aid2uids[config.aid])
    basic_shape = (None, config['sample_size'], n_units) \
        if config.get('sample_size') else (None, n_units)
    shapes = env_stats['obs_shape'][config.aid]
    dtypes = env_stats['obs_dtype'][config.aid]
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}

    data_format.update(dict(
        action=((*basic_shape, *env_stats.action_shape), env_stats.action_dtype, 'action'),
        plogits=((*basic_shape, env_stats.action_dim), tf.float32, 'plogits'),
        paction=((*basic_shape, *env_stats.action_shape), env_stats.action_dtype, 'paction'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logpi=(basic_shape, tf.float32, 'logpi'),
    ))

    if config.get('store_state'):
        data_format['mask'] = (basic_shape, tf.float32, 'mask')
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        state_type = type(model.state_size)
        data_format['state'] = state_type(*[((None, sz), dtype, name) 
            for name, sz in model.state_size._asdict().items()])
    
    return data_format
