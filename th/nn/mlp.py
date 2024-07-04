import torch
from torch import nn
from torch.utils._pytree import tree_map

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath('__file__'))))

from tools.log import do_logging
from th.utils import tpdv
from th.nn.layers import RNNLayer, LSTMState
from th.nn.registry import nn_registry
from th.nn.utils import get_initializer, get_activation, calculate_scale


def _prepare_for_rnn(x):
  if x is None:
    return x, None
  x = x.transpose(0, 1)
  shape = x.shape
  x = x.reshape(x.shape[0], -1, *x.shape[3:])
  return x, shape


def _recover_shape(x, shape):
  x = x.reshape(*shape[:3], x.shape[-1])
  x = x.transpose(0, 1)
  return x


def _prepare_rnn_state(state, shape):
  state = tree_map(
    lambda x: x.permute(2, 0, 1, 3).reshape(shape), state
  )
  return state


def _recover_rnn_state(state, shape):
  state = tree_map(
    lambda x: x.permute(1, 0, 2).reshape(shape), state
  )
  return state

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
    norm=None, 
    norm_after_activation=False, 
    norm_kwargs={
      'elementwise_affine': True, 
    }, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    rnn_type=None, 
    rnn_layers=1, 
    rnn_units=None, 
    rnn_init='orthogonal',
    rnn_norm=False, 
  ):
    super().__init__()
    if activation is None and (len(units_list) > 1 or (units_list and out_size)):
      do_logging(f'MLP({name}) with units_list({units_list}) and out_size({out_size}) has no activation.', color='red')

    gain = calculate_scale(activation)
    w_init = get_initializer(w_init, gain=gain)
    b_init = get_initializer(b_init)
    units_list = [input_dim] + units_list
    self.layers = nn.Sequential()
    for i, u in enumerate(units_list[1:]):
      layers = nn.Sequential()
      l = nn.Linear(units_list[i], u)
      w_init(l.weight.data)
      b_init(l.bias.data)
      layers.append(l)
      if norm == 'layer' and not norm_after_activation:
        layers.append(nn.LayerNorm(u, **norm_kwargs))
      layers.append(get_activation(activation))
      if norm == 'layer' and norm_after_activation:
        layers.append(nn.LayerNorm(u, **norm_kwargs))
      self.layers.append(layers)

    self.rnn_type = rnn_type
    self.rnn_layers = rnn_layers
    self.rnn_units = rnn_units
    if rnn_type is None:
      self.rnn = None
      input_dim = u
    else:
      self.rnn = RNNLayer(
        u, rnn_units, rnn_type, 
        rnn_layers=rnn_layers, 
        rnn_init=rnn_init, 
        rnn_norm=rnn_norm
      )
      input_dim = rnn_units
    
    if out_size is not None:
      self.out_layer = nn.Linear(input_dim, out_size)
      w_init = get_initializer(out_w_init, gain=out_scale)
      b_init = get_initializer(out_b_init)
      w_init(self.out_layer.weight.data)
      b_init(self.out_layer.bias.data)
    else:
      self.out_layer = None

  def forward(self, x, reset=None, state=None):
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
        if self.rnn_type == 'lstm':
          state = LSTMState(
            torch.zeros(shape[1], shape[2], self.rnn_layers, self.rnn_units).to(device=x.device), 
            torch.zeros(shape[1], shape[2], self.rnn_layers, self.rnn_units).to(device=x.device)
          )
        else:
          state = torch.zeros(shape[1], shape[2], self.rnn_layers, self.rnn_units).to(device=x.device)
      state = _prepare_rnn_state(state, (self.rnn_layers, shape[1] * shape[2], self.rnn_units))
      x, state = self.rnn(x, state, reset)
      x = _recover_shape(x, shape)
      state = _recover_rnn_state(state, (shape[1], shape[2], self.rnn_layers, self.rnn_units))
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
    units_list=[64, 64],
    w_init='orthogonal',
    activation='relu', 
    norm='layer',
    norm_after_activation=True,
    out_scale=.01,
    rnn_type='lstm',
    rnn_units=64,
    rnn_init=None,
    rnn_norm='layer',
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
