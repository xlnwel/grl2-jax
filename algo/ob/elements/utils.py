import tensorflow as tf


def get_data_format(config, env_stats, model):
    aid = config.aid
    n_units = len(env_stats.aid2uids[aid])
    basic_shape = (None, config['sample_size'], n_units) \
        if model.has_rnn else (None, n_units)
    shapes = env_stats['obs_shape'][aid]
    dtypes = env_stats['obs_dtype'][aid]
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}

    action_shape = env_stats.action_shape[aid]
    action_dtype = env_stats.action_dtype[aid]
    data_format.update(dict(
        action=((*basic_shape, *action_shape), action_dtype, 'action'),
        reward=(basic_shape, tf.float32, 'reward'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        raw_adv=(basic_shape, tf.float32, 'raw_adv'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logpi=(basic_shape, tf.float32, 'logpi'),
    ))

    if config.get('store_state'):
        if config.get('actor_rnn_type') or config.get('value_rnn_type'):
            data_format['mask'] = (basic_shape, tf.float32, 'mask')
            dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        if config.get('actor_rnn_type'):
            data_format['actor_state'] = model.actor_state_type(*[((None, sz), dtype, name) 
                for name, sz in model.actor_state_size._asdict().items()])
        if config.get('value_rnn_type'):
            data_format['value_state'] = model.value_state_type(*[((None, sz), dtype, name) 
                for name, sz in model.value_state_size._asdict().items()])

    return data_format
