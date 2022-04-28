import tensorflow as tf

from utility.typing import AttrDict
from algo.hm.elements.utils import get_basics, \
    update_data_format_with_rnn_states


def get_data_format(
    config: AttrDict, 
    env_stats: AttrDict, 
    model
):
    data_format, basic_shape, action_shape, \
        action_dim, action_dtype = \
        get_basics(config, env_stats, model)

    data_format.update(dict(
        action=((*basic_shape, *action_shape), action_dtype, 'action'),
        reward=(basic_shape, tf.float32, 'reward'), 
        value=(basic_shape, tf.float32, 'value'), 
        value_a=(basic_shape, tf.float32, 'value_a'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        traj_ret_a=(basic_shape, tf.float32, 'traj_ret_a'),
        raw_adv=(basic_shape, tf.float32, 'raw_adv'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        target_prob=(basic_shape, tf.float32, 'target_prob'),
        tr_prob=(basic_shape, tf.float32, 'tr_prob'),
        logprob=(basic_shape, tf.float32, 'logprob'),
    ))
    if env_stats.is_action_discrete:
        data_format['pi'] = ((*basic_shape, action_dim), tf.float32, 'pi')
    else:
        data_format['pi_mean'] = ((*basic_shape, action_dim), tf.float32, 'pi_mean')
        data_format['pi_std'] = ((*basic_shape, action_dim), tf.float32, 'pi_std')

    data_format = update_data_format_with_rnn_states(
        data_format,
        config,
        basic_shape,
        model.policy,
        model.value,
    )

    return data_format
