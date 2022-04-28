import tensorflow as tf


def get_data_format(config, env_stats, model):
    basic_shape = (None, config['sample_size']) \
        if hasattr(model, 'rnn') else (None,)
    shapes = env_stats['obs_shape']
    dtypes = env_stats['obs_dtype']
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}

    data_format.update(dict(
        action=((*basic_shape, *env_stats['action_shape']), env_stats['action_dtype'], 'action'),
        reward=(basic_shape, tf.float32, 'reward'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        raw_adv=(basic_shape, tf.float32, 'raw_adv'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logprob=(basic_shape, tf.float32, 'logprob'),
    ))

    if config.get('store_state'):
        assert model.state_size is not None, model.state_size
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        state_type = type(model.state_size)
        data_format['state'] = state_type(*[((None, sz), dtype, name) 
            for name, sz in model.state_size._asdict().items()])
    
    return data_format

def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    buffer.add(**kwargs)
