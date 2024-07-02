from jax import lax, nn
import jax.numpy as jnp
import haiku as hk

from core.names import DEFAULT_ACTION
from core.typing import dict2AttrDict
from nn.func import nn_registry
from nn.layers import Layer
from nn.mlp import MLP
from nn.utils import get_activation
from jax_tools import jax_assert
""" Source this file to register Networks """


@nn_registry.register('policy')
class Policy(hk.Module):
  def __init__(
    self, 
    is_action_discrete, 
    action_dim, 
    out_act=None, 
    init_std=.2, 
    sigmoid_scale=True, 
    std_x_coef=1., 
    std_y_coef=.5, 
    use_action_mask={DEFAULT_ACTION: False}, 
    use_feature_norm=False, 
    name='policy', 
    **config
  ):
    super().__init__(name=name)
    self.config = dict2AttrDict(config, to_copy=True)
    self.action_dim = action_dim
    self.is_action_discrete = is_action_discrete

    self.out_act = out_act
    self.init_std = init_std
    self.sigmoid_scale = sigmoid_scale
    self.std_x_coef = std_x_coef
    self.std_y_coef = std_y_coef
    self.use_action_mask = use_action_mask
    self.use_feature_norm = use_feature_norm

  def __call__(self, x, reset=None, state=None, prev_info=None, action_mask=None, no_state_return=False):
    if self.use_feature_norm:
      ln = hk.LayerNorm(-1, True, True)
      x = ln(x)
    net, heads = self.build_net()
    x = net(x, reset, state=state, prev_info=prev_info)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    
    if action_mask is not None and not isinstance(action_mask, dict):
      assert len(heads) == 1, heads
      action_mask = {k: action_mask for k in heads}

    outs = {}
    for name, layer in heads.items():
      v = layer(x)
      if self.is_action_discrete[name]:
        if self.use_action_mask.get(name, False):
          assert action_mask[name] is not None, action_mask
          am = action_mask[name]
          jax_assert.assert_shape_compatibility([v, am])
          v = jnp.where(am, v, -float('inf'))
          v = jnp.where(jnp.all(am == 0, axis=-1, keepdims=True), 1, v)
        outs[name] = v
      else:
        if self.out_act == 'tanh':
          v = jnp.tanh(v)
        if self.sigmoid_scale:
          logstd_init = self.std_x_coef
        else:
          logstd_init = lax.log(self.init_std)
        logstd_init = hk.initializers.Constant(logstd_init)
        logstd = hk.get_parameter(
          f'{name}_logstd', 
          shape=(self.action_dim[name],), 
          init=logstd_init
        )
        if self.sigmoid_scale:
          scale = nn.sigmoid(logstd / self.std_x_coef) * self.std_y_coef
        else:
          scale = lax.exp(logstd)
        outs[name] = (v, scale)
    if no_state_return:
      return outs
    else:
      return outs, state

  @hk.transparent
  def build_net(self):
    net = MLP(**self.config, name=self.name)
    out_kwargs = net.out_kwargs
    if isinstance(self.action_dim, dict):
      heads = {k: Layer(v, **out_kwargs, name=f'head_{k}') 
               for k, v in self.action_dim.items()}
    else:
      raise NotImplementedError(self.action_dim)

    return net, heads


@nn_registry.register('value')
class Value(hk.Module):
  def __init__(
    self, 
    out_act=None, 
    out_size=1, 
    name='value', 
    use_feature_norm=False, 
    **config
  ):
    super().__init__(name=name)
    self.config = dict2AttrDict(config, to_copy=True)
    self.out_act = get_activation(out_act)
    self.out_size = out_size
    self.use_feature_norm = use_feature_norm

  def __call__(self, x, reset=None, state=None, prev_info=None):
    if self.use_feature_norm:
      ln = hk.LayerNorm(-1, True, True)
      x = ln(x)
    net = self.build_net()
    x = net(x, reset, state=state, prev_info=prev_info)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x

    if x.shape[-1] == 1:
      x = jnp.squeeze(x, -1)
    value = self.out_act(x)

    return value, state

  @hk.transparent
  def build_net(self):
    net = MLP(
      **self.config, 
      out_size=self.out_size, 
      name=self.name
    )

    return net


if __name__ == '__main__':
  import jax
  # config = dict( 
  #   w_init='orthogonal', 
  #   scale=1, 
  #   activation='relu', 
  #   norm='layer', 
  #   out_scale=.01, 
  #   out_size=2
  # )
  # def layer_fn(x, *args):
  #   layer = HyperParamEmbed(**config)
  #   return layer(x, *args)
  # import jax
  # rng = jax.random.PRNGKey(42)
  # x = jax.random.normal(rng, (2, 3))
  # net = hk.transform(layer_fn)
  # params = net.init(rng, x, 1, 2, 3.)
  # print(params)
  # print(net.apply(params, None, x, 1., 2, 3))
  # print(hk.experimental.tabulate(net)(x, 1, 2, 3.))
  import os, sys
  os.environ["XLA_FLAGS"] = '--xla_dump_to=/tmp/foo'
  os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

  config = {
    'units_list': [3], 
    'w_init': 'orthogonal', 
    'activation': 'relu', 
    'norm': None, 
    'out_scale': .01,
  }
  def layer_fn(x, *args):
    layer = EnsembleModels(5, 3, **config)
    return layer(x, *args)
  import jax
  rng = jax.random.PRNGKey(42)
  x = jax.random.normal(rng, (2, 3, 3))
  a = jax.random.normal(rng, (2, 3, 2))
  net = hk.transform(layer_fn)
  params = net.init(rng, x, a)
  print(params)
  print(net.apply(params, rng, x, a))
  print(hk.experimental.tabulate(net)(x, a))

  # config = {
  #   'units_list': [64,64], 
  #   'w_init': 'orthogonal', 
  #   'activation': 'relu', 
  #   'norm': None, 
  #   'index': 'all', 
  #   'index_config': {
  #     'use_shared_bias': False, 
  #     'use_bias': True, 
  #     'w_init': 'orthogonal', 
  #   }
  # }
  # def net_fn(x, *args):
  #   net = Value(**config)
  #   return net(x, *args)

  # rng = jax.random.PRNGKey(42)
  # x = jax.random.normal(rng, (2, 3, 4))
  # hx = jnp.eye(3)
  # hx = jnp.tile(hx, [2, 1, 1])
  # net = hk.transform(net_fn)
  # params = net.init(rng, x, hx)
  # print(params)
  # print(net.apply(params, rng, x, hx))
  # print(hk.experimental.tabulate(net)(x, hx))

  # config = {
  #   'units_list': [2, 3], 
  #   'w_init': 'orthogonal', 
  #   'activation': 'relu', 
  #   'norm': None, 
  #   'out_scale': .01,
  #   'rescale': .1, 
  #   'out_act': 'atan', 
  #   'combine_xa': True, 
  #   'out_size': 3, 
  #   'index': 'all', 
  #   'index_config': {
  #     'use_shared_bias': False, 
  #     'use_bias': True, 
  #     'w_init': 'orthogonal', 
  #   }
  # }
  # def net_fn(x, *args):
  #   net = Reward(**config)
  #   return net(x, *args)

  # rng = jax.random.PRNGKey(42)
  # x = jax.random.normal(rng, (2, 3, 4))
  # action = jax.random.randint(rng, (2, 3), minval=0, maxval=3)
  # hx = jnp.eye(3)
  # hx = jnp.tile(hx, [2, 1, 1])
  # net = hk.transform(net_fn)
  # params = net.init(rng, x, action, hx)
  # print(params)
  # print(net.apply(params, rng, x, action, hx))
  # print(hk.experimental.tabulate(net)(x, action, hx))
