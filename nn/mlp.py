import jax
import jax.numpy as jnp
import haiku as hk

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath('__file__'))))

from core.log import do_logging
from nn.layers import Layer
from nn.registry import layer_registry, nn_registry
from nn.utils import call_norm


def _prepare_for_rnn(x):
  x = jnp.swapaxes(x, 0, 1)
  shape = x.shape
  x = jnp.reshape(x, (x.shape[0], -1, *x.shape[3:]))
  return x, shape

def _recover_shape(x, shape):
  x = jnp.reshape(x, (*shape[:3], x.shape[-1]))
  x = jnp.swapaxes(x, 0, 1)
  return x

def _rnn_reshape(rnn_out, shape):
  rnn_out = jax.tree_util.tree_map(lambda x: x.reshape(shape), rnn_out)
  return rnn_out


@nn_registry.register('mlp')
class MLP(hk.Module):
  def __init__(
    self, 
    units_list=[], 
    out_size=None, 
    layer_type='linear', 
    norm=None, 
    activation=None, 
    w_init='glorot_uniform', 
    b_init='zeros', 
    name=None, 
    out_scale=1, 
    norm_after_activation=False, 
    norm_kwargs={
      'axis': -1, 
      'create_scale': True, 
      'create_offset': True, 
    }, 
    out_layer_type=None, 
    out_w_init=None, 
    out_b_init=None, 
    rnn_type=None, 
    rnn_units=None, 
    rnn_norm=None, 
    out_kwargs={}, 
    **kwargs
  ):
    super().__init__(name=name)
    if activation is None and (len(units_list) > 1 or (units_list and out_size)):
      do_logging(f'MLP({name}) with units_list({units_list}) and out_size({out_size}) has no activation.', color='red')

    self.units_list = units_list
    self.layer_kwargs = dict(
      layer_type=layer_type, 
      norm=norm, 
      activation=activation, 
      w_init=w_init, 
      norm_after_activation=norm_after_activation, 
      norm_kwargs=norm_kwargs, 
      **kwargs
    )

    self.out_size = out_size
    do_logging(f'{self.name} out scale: {out_scale}')
    if out_layer_type is None:
      out_layer_type = layer_type
    if out_w_init is None:
      out_w_init = w_init
    if out_b_init is None:
      out_b_init = b_init
    self.norm = norm
    self.norm_kwargs = norm_kwargs
    self.rnn_norm = rnn_norm
    self.out_kwargs = dict(
      layer_type=out_layer_type, 
      w_init=out_w_init, 
      b_init=out_b_init, 
      scale=out_scale, 
      **out_kwargs
    )

    self.rnn_type = rnn_type
    assert self.rnn_type in (None, 'gru', 'lstm'), self.rnn_type
    self.rnn_units = rnn_units

  def __call__(self, x, reset=None, state=None, prev_info=None, is_training=True):
    if self.rnn_type is None:
      layers = self.build_net()
      for l in layers:
        x = l(x, is_training)
      return x
    else:
      layers, core, out_layers = self.build_net()
      for l in layers:
        x = l(x, is_training)
      if state is None:
        state = core.initial_state(x.shape[0] * x.shape[2])
      
      # we assume the original data is of form [B, T, U, *]
      if prev_info is not None:
        x = jnp.concatenate([x, prev_info], -1)
      x, shape = _prepare_for_rnn(x)
      reset, _ = _prepare_for_rnn(reset)
      x = (x, reset)
      state = _rnn_reshape(state, (shape[1] * shape[2], -1))
      x, state = hk.dynamic_unroll(core, x, state)
      x = _recover_shape(x, shape)
      state = _rnn_reshape(state, (shape[1], shape[2], -1))
      x = call_norm(x, self.rnn_norm, self.norm_kwargs, is_training=is_training, name='rnn_norm')

      for l in out_layers:
        x = l(x, is_training)
      return x, state
  
  @hk.transparent
  def build_net(self):
    if self.rnn_type is None:
      layers = []
      for u in self.units_list:
        layers.append(Layer(u, **self.layer_kwargs))
      if self.out_size:
        layers.append(Layer(self.out_size, **self.out_kwargs, name='out'))

      return layers
    else:
      assert isinstance(self.rnn_units, int), self.rnn_units
      layers = []
      for u in self.units_list:
        layers.append(Layer(u, **self.layer_kwargs))

      if self.rnn_type is not None:
        if self.rnn_type == 'lstm':
          core = hk.LSTM(self.rnn_units)
        elif self.rnn_type == 'gru':
          core = hk.GRU(self.rnn_units)
        core = hk.ResetCore(core)

      out_layers = []
      if self.out_size:
        out_layers.append(Layer(self.out_size, **self.out_kwargs, name='out'))

      return layers, core, out_layers


if __name__ == '__main__':
  config = dict(
    units_list=[2, 3], 
    w_init='orthogonal', 
    scale=1, 
    activation='relu', 
    norm='layer', 
    name='mlp', 
    out_scale=.01, 
    out_size=1, 
    rnn_type='gru', 
    rnn_units=2
  )
  def mlp(x, reset=None, state=None):
    layer = MLP(**config)
    return layer(x, reset, state)
  import jax.numpy as jnp
  rng = jax.random.PRNGKey(42)
  b = 2
  s = 3
  d = 4
  # x = jax.random.normal(rng, (b, s, d))
  x = jnp.ones((b, s, 1, d))
  reset = jnp.ones((b, s, 1))
  net = hk.transform(mlp)
  params = net.init(rng, x, reset)
  out, state = net.apply(params, rng, x, reset)
  print('first x', out)
  state = jax.tree_util.tree_map(jnp.ones_like, state)
  reset = jnp.ones((b, s))
  out, state = net.apply(params, rng, x, reset, state)
  # print('next x', out)
  print(state)
  # print(hk.experimental.tabulate(net)(x, reset))