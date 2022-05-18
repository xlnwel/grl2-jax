import tensorflow as tf


def get_data_format(config, env_stats, model):
    basic_shape = (None, config['sample_size']) \
        if hasattr(model, 'rnn') else (None,)
    shapes = env_stats.obs_shape
    dtypes = env_stats.obs_dtype
    action_dim = env_stats.action_dim
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}

    data_format.update(dict(
        action=((*basic_shape, *env_stats['action_shape']), env_stats['action_dtype'], 'action'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        target_prob=(basic_shape, tf.float32, 'target_prob'),
        tr_prob=(basic_shape, tf.float32, 'tr_prob'),
        vt_prob=(basic_shape, tf.float32, 'vt_prob'),
        target_prob_prime=(basic_shape, tf.float32, 'target_prob_prime'),
        tr_prob_prime=(basic_shape, tf.float32, 'tr_prob_prime'),
        logprob=(basic_shape, tf.float32, 'logprob'),
    ))
    if env_stats.is_action_discrete:
        data_format['pi'] = ((*basic_shape, action_dim), tf.float32, 'pi')
        data_format['target_pi'] = ((*basic_shape, action_dim), tf.float32, 'target_pi')
    else:
        data_format['pi_mean'] = ((*basic_shape, action_dim), tf.float32, 'pi_mean')
        data_format['pi_std'] = ((*basic_shape, action_dim), tf.float32, 'pi_std')
    

    if config.get('store_state') and config.get('rnn_type'):
        assert model.state_size is not None, model.state_size
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        state_type = type(model.state_size)
        data_format['mask'] = (basic_shape, tf.float32, 'mask')
        data_format['state'] = state_type(*[((None, sz), dtype, name) 
            for name, sz in model.state_size._asdict().items()])
    
    return data_format

def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    buffer.add(**kwargs)
