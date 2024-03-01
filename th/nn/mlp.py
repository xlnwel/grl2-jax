import logging
from torch import nn
from torch.utils._pytree import tree_map

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath('__file__'))))

from core.log import do_logging
from th.nn.layers import RNNLayer
from th.nn.registry import nn_registry
from th.nn.utils import get_initializer, get_activation, calculate_scale

logger = logging.getLogger(__name__)


def _prepare_for_rnn(x):
  x = x.transpose(0, 1)
  shape = x.shape
  x = x.reshape(x.shape[0], -1, *x.shape[3:])
  return x, shape

def _recover_shape(x, shape):
  x = x.reshape(*shape[:3], x.shape[-1])
  x = x.transpose(0, 1)
  return x

def _rnn_reshape(rnn_out, shape):
  rnn_out = tree_map(lambda x: x.reshape(shape), rnn_out)
  return rnn_out


@nn_registry.register('mlp')
class MLP(nn.Module):
  def __init__(
    self, 
    input_dim, 
    units_list=[], 
    out_size=None, 
    activation=None, 
    w_init='glorot_uniform', 
    b_init='zeros', 
    name=None, 
    out_scale=1, 
    norm_after_activation=False, 
    norm_kwargs={
      'elementwise_affine': True, 
      'bias': True, 
    }, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    rnn_type=None, 
    rnn_layers=1, 
    rnn_units=None, 
    rnn_norm=False, 
  ):
    super().__init__()
    if activation is None and (len(units_list) > 1 or (units_list and out_size)):
      do_logging(f'MLP({name}) with units_list({units_list}) and out_size({out_size}) has no activation.', 
        logger=logger, level='pwc')

    w_init = get_initializer(w_init)
    gain = calculate_scale(activation)
    b_init = get_initializer(b_init)
    units_list = [input_dim] + units_list
    self.layers = nn.Sequential()
    for i, u in enumerate(units_list[1:]):
      l = nn.Linear(units_list[i], u)
      w_init(l.weight.data, gain=gain)
      b_init(l.bias.data)
      self.layers.append(l)
      if not norm_after_activation:
        self.layers.append(nn.LayerNorm(u, **norm_kwargs))
      self.layers.append(get_activation(activation))
      if norm_after_activation:
        self.layers.append(nn.LayerNorm(u, **norm_kwargs))

    self.rnn_units = rnn_units
    if rnn_type is None:
      self.rnn = None
      input_dim = u
    else:
      self.rnn = RNNLayer(u, rnn_units, rnn_layers=rnn_layers, rnn_norm=rnn_norm)
      input_dim = rnn_units
    
    if out_size is not None:
      self.out_layer = nn.Linear(input_dim, out_size)
      w_init = get_initializer(out_w_init)
      b_init = get_initializer(out_b_init)
      w_init(self.out_layer.weight.data, gain=out_scale)
      b_init(self.out_layer.bias.data)
    else:
      self.out_layer = None

  def __call__(self, x, reset=None, state=None, is_training=True):
    if self.rnn is None:
      x = self.layers(x)
      if self.out_layer is not None:
        x = self.out_layer(x)
      return x
    else:
      x = self.layers(x)
      x, shape = _prepare_for_rnn(x)
      reset, _ = _prepare_for_rnn(reset)
      if state is None:
        state = torch.zeros(1, shape[1], shape[2], self.rnn_units)
      state = _rnn_reshape(state, (shape[1] * shape[2], -1))
      x, state = self.rnn(x, state, 1-reset.float())
      x = _recover_shape(x, shape)
      state = _rnn_reshape(state, (shape[1], shape[2], -1))
      if self.out_layer is not None:
        x = self.out_layer(x)
      return x, state


if __name__ == '__main__':
  b = 4
  s = 3
  u = 2
  d = 5
  config = dict(
    input_dim=d, 
    units_list=[2, 3], 
    w_init='orthogonal', 
    activation='relu', 
    norm_after_activation=True,
    out_scale=.01, 
    out_size=1, 
    rnn_type='gru', 
    rnn_units=2
  )
  import torch
  mlp = MLP(**config)
  print(mlp)
  x = torch.rand(b, s, u, d)
  reset = torch.randn(b, s, u) < .5
  # state = torch.zeros(1, b, u, d)
  x, state = mlp(x, reset)
  print('output:', x.shape)
  print('state:', state.shape)
