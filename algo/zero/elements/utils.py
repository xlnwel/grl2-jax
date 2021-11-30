import tensorflow as tf


def get_data_format(config, env_stats, model, use_for_dataset=True):
    if config['training'] == 'ppo':
        return get_ppo_data_format(config, env_stats, model, use_for_dataset)
    else:
        return get_bc_data_format(config, env_stats, model, use_for_dataset)


def get_ppo_data_format(config, env_stats, model, use_for_dataset=True):
    basic_shape = (None, config['sample_size'])
    shapes = {**env_stats['obs_shape'], **env_stats['action_shape']}
    dtypes = {**env_stats['obs_dtype'], **env_stats['action_dtype']}
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}
    
    data_format.update(dict(
        # card_rank_mask=((*basic_shape, env_stats['obs_shape']['card_rank_mask'][-1]), tf.bool, 'card_rank_mask'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        action_type_logpi=(basic_shape, tf.float32, 'action_type_logpi'),
        card_rank_logpi=(basic_shape, tf.float32, 'card_rank_logpi'),
    ))

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


def get_bc_data_format(config, env_stats, model, use_for_dataset=True):
    basic_shape = (None, config['sample_size'])
    shapes = {**env_stats['obs_shape'], **env_stats['action_shape']}
    dtypes = {**env_stats['obs_dtype'], **env_stats['action_dtype']}
    data_format = {k: ((*basic_shape, *v), dtypes[k], k) 
        for k, v in shapes.items()}

    data_format.update(dict(
        mask=(basic_shape, tf.float32, 'mask'),
        # reward=(basic_shape, tf.float32, 'reward'),
    ))

    return data_format


def collect(buffer, env, env_step, reset, obs, action, next_obs, **kwargs):
    # obs['card_rank_mask'] = card_rank_mask
    kwargs['action_type'], kwargs['card_rank'] = action
    buffer.add(**obs, **kwargs)
