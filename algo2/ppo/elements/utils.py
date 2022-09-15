import functools
import tensorflow as tf

from algo.gpo.elements.utils import collect, get_data_format as get_data_format_base


def update_data_format_with_rnn_states(
    data_format, 
    config, 
    basic_shape, 
    model
):
    if config.get('store_state') and config.get('rnn_type'):
        assert model.state_size is not None, model.state_size
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        state_type = type(model.state_size)
        data_format['mask'] = (basic_shape, tf.float32, 'mask')
        data_format['state'] = state_type(*[((None, sz), dtype, name) 
            for name, sz in model.state_size._asdict().items()])
    
    return data_format

get_data_format = functools.partial(
    get_data_format_base, rnn_state_fn=update_data_format_with_rnn_states)