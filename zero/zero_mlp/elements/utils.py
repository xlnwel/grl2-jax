import tensorflow as tf

from utility.typing import AttrDict
from algo.gpo.elements.utils import get_basics, \
    get_data_format as get_ppo_data_format


def get_data_format(
    config: AttrDict, 
    env_stats: AttrDict, 
    model
):
    basic_shape, shapes, dtypes, action_shape, \
        action_dim, action_dtype = \
        get_basics(config, env_stats, model)
    
    data_format = get_ppo_data_format(config, env_stats, model)

    data_format.update(dict(
        value_a=(basic_shape, tf.float32, 'value_a'),
        traj_ret_a=(basic_shape, tf.float32, 'traj_ret_a'),
    ))

    return data_format
