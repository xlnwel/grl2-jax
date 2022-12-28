import logging
import jax
import jax.numpy as jnp
import haiku as hk
from haiku import initializers

from core.log import do_logging
from nn.dummy import Dummy

logger = logging.getLogger(__name__)


def get_activation(act_name, **kwargs):
    activations = {
        None: Dummy(),
        'relu': jax.nn.relu,
        'leaky_relu': jax.nn.leaky_relu,
        'elu': jax.nn.elu,
        'sigmoid': jax.nn.sigmoid, 
        'tanh': jnp.tanh, 
        'atan': jax.lax.atan, 
    }
    if isinstance(act_name, str):
        act_name = act_name.lower()
    assert act_name in activations, act_name
    return activations[act_name]


@hk.transparent
def get_norm(name):
    norm_layers = {
        None: Dummy,
        'layer': hk.LayerNorm,
        'batch': hk.BatchNorm,
    }
    """ Return a normalization """
    if isinstance(name, str):
        name = name.lower()
    if name in norm_layers:
        return norm_layers[name]
    else:
        # assume name is an normalization layer class
        return name


def calculate_scale(name, param=None):
    """ a jax replica of torch.nn.init.calculate_gain """
    m = {
        None: 1, 
        'sigmoid': 1, 
        'tanh': 5./3., 
        'relu': jnp.sqrt(2.), 
        'leaky_relu': jnp.sqrt(2./(1+(param or 0)**2)),
    }
    return m[name]


def get_initializer(name, **kwargs):
    """ 
    Return a parameter initializer by name
    """
    scale = kwargs.get('scale', 1.)
    inits = {
        'orthogonal': initializers.Orthogonal(scale), 
        'glorot_uniform': initializers.VarianceScaling(1., 'fan_avg', 'uniform'), 
        'glorot_normal': initializers.VarianceScaling(1., 'fan_avg', 'truncated_normal'), 
        'he_uniform': initializers.VarianceScaling(2., 'fan_in', 'uniform'), 
        'he_normal': initializers.VarianceScaling(2., 'fan_in', 'truncated_normal'), 
        'truncated_normal': initializers.TruncatedNormal(), 
        'zeros': initializers.Constant(0), 
    }
    if isinstance(name, str):
        name = name.lower()
        if name in inits:
            return inits[name]
        elif name.startswith('const'):
            val = float(name.split('_')[-1])
            act = initializers.Constant(val)
            return act
        else:
            ValueError(f'Unknonw initializer: {name}')
    else:
        return name


@hk.transparent
def call_norm(norm_type, norm_kwargs, x, is_training, name=None):
    if norm_type is None:
        return x
    norm_layer = get_norm(norm_type)(**norm_kwargs, name=name)
    if norm_type == 'batch':
        y = norm_layer(x, is_training=is_training)
    else:
        y = norm_layer(x)
    return y
