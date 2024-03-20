import logging
import numpy as np
from torch import nn

from core.log import do_logging
from nn.dummy import Dummy

logger = logging.getLogger(__name__)


def get_activation(act_name, **kwargs):
  activations = {
    None: Dummy(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
  }
  if isinstance(act_name, str):
    act_name = act_name.lower()
  assert act_name in activations, act_name
  return activations[act_name]


def get_norm(name):
  norm_layers = {
    None: Dummy,
    'layer': nn.LayerNorm,
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
    'relu': np.sqrt(2.), 
    'leaky_relu': np.sqrt(2./(1+(param or 0)**2)),
  }
  return m.get(name, 1)


def init_linear(module, w_init, b_init, scale):
  w_init = get_initializer(w_init)
  b_init = get_initializer(b_init)
  w_init(module.weight.data, gain=scale)
  b_init(module.bias.data)
  return module

def get_initializer(name, **kwargs):
  """ 
  Return a parameter initializer by name
  """
  inits = {
    'orthogonal': nn.init.orthogonal_, 
    'glorot_uniform': nn.init.xavier_uniform_, 
    'glorot_normal': nn.init.xavier_normal_, 
    'he_uniform': nn.init.kaiming_uniform_, 
    'he_normal': nn.init.kaiming_normal_, 
    'truncated_normal': nn.init.trunc_normal_, 
    'zeros': nn.init.zeros_, 
  }
  if isinstance(name, str):
    name = name.lower()
    if name in inits:
      return inits[name]
    elif name.startswith('const'):
      val = float(name.split('_')[-1])
      act = lambda x: nn.init.constant_(x, val)
      return act
    else:
      ValueError(f'Unknonw initializer: {name}')
  else:
    return name


def reset_weights(weights, rng, name, **params):
  nn.init.orthogonal_(weights.weight.data, **params)
  if weights.bias is not None:
    nn.init.zeros_(weights.bias.data)
  return weights

