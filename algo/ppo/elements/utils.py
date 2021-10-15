import tensorflow as tf


def get_data_format(config, env_stats, model, use_for_dataset=True):
    basic_shape = (None, config['sample_size']) \
        if hasattr(model, 'rnn') else (None,)
    data_format = dict(
        obs=((*basic_shape, *env_stats.obs_shape), env_stats.obs_dtype, 'obs'),
        action=((*basic_shape, *env_stats.action_shape), env_stats.action_dtype, 'action'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logpi=(basic_shape, tf.float32, 'logpi'),
    )

    if config.get('store_state'):
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        if use_for_dataset:
            data_format.update({
                name: ((None, sz), dtype)
                    for name, sz in model.state_size._asdict().items()
            })
        else:
            state_type = type(model.state_size)
            data_format['state'] = state_type(*[((None, sz), dtype, name) 
                for name, sz in model.state_size._asdict().items()])
    
    return data_format

def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    kwargs['reset'] = reset
    buffer.add(**kwargs)
