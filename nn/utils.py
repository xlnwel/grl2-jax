import logging
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, initializers
from tensorflow.keras.mixed_precision.experimental import global_policy # useful for modules import nn.utils

logger = logging.getLogger(__name__)

custom_activations = dict(
    leaky_relu=tf.nn.leaky_relu,
    hswish=lambda x: x * (tf.nn.relu6(x+3) / 6) # see MobileNet3
)

def get_activation(name):
    if isinstance(name, str):
        name = name.lower()
    if name is None:
        return None
    elif name in custom_activations:
        return custom_activations[name]
    else:
        return activations.get(name)
        
def get_norm(name):
    """ Return a normalization """
    if isinstance(name, str):
        if name == 'layer':
            return layers.LayerNormalization
        elif name == 'batch':
            return layers.BatchNormalization
        elif name.lower():
            return None
        else:
            raise NotImplementedError
    else:
        # assume name is an normalization layer instance
        return name

def calculate_gain(name, param=None):
    """ a replica of torch.nn.init.calculate_gain """
    m = {
        None: 1, 
        'sigmoid': 1, 
        'tanh': 5./3., 
        'relu': np.sqrt(2.), 
        'leaky_relu': np.sqrt(2./(1+(param or 0)**2)),
        # the followings are I make up
        'elu': np.sqrt(2.),
        'hswish': np.sqrt(2.), 
    }
    return m[name]

def constant_initializer(val):
    return initializers.Constant(val)

def get_initializer(name, **kwargs):
    """ 
    Return a kernel initializer by name
    """
    if isinstance(name, str):
        if name.lower() == 'none':
            return None
        elif name.lower() == 'orthogonal':
            gain = kwargs.get('gain', np.sqrt(2.))
            return initializers.orthogonal(gain)
        return initializers.get(name)
    else:
        return name

def ortho_init(scale=1.0):
    """ 
    A reproduction of tf...Orthogonal, originally from openAI baselines
    """
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def convert_obs(x, obs_range, dtype=tf.float32):
    if x.dtype != np.uint8:
        logger.debug(f'Observations({x.shape}, {x.dtype}) are already converted to {x.dtype}, no further process is performed')
        return x
    dtype = dtype or tf.float32 # dtype is None when global policy is not unspecified, override it
    logger.debug(f'Observations({x.shape}, {x.dtype}) are converted to range {obs_range} of dtype {dtype}')
    if obs_range == [0, 1]:
        return tf.cast(x, dtype) / 255.
    elif obs_range == [-.5, .5]:
        return tf.cast(x, dtype) / 255. - .5
    elif obs_range == [-1, 1]:
        return tf.cast(x, dtype) / 127.5 - 1.
    else:
        raise ValueError(obs_range)

def flatten(x):
    shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
    x = tf.reshape(x, shape)
    return x