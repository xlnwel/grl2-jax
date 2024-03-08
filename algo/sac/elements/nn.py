from jax import lax, nn
import jax.numpy as jnp
import haiku as hk

from core.names import DEFAULT_ACTION
from core.typing import dict2AttrDict
from nn.func import nn_registry
from nn.layers import Layer
from nn.mlp import MLP
from jax_tools import jax_assert


@nn_registry.register('policy')
class Policy(hk.Module):
  def __init__(
    self, 
    is_action_discrete, 
    action_dim, 
    LOG_STD_MIN=-20, 
    LOG_STD_MAX=2, 
    use_feature_norm=False, 
    name='policy', 
    **config
  ):
    super().__init__(name=name)
    self.config = dict2AttrDict(config, to_copy=True)
    self.action_dim = action_dim
    self.is_action_discrete = is_action_discrete

    self.LOG_STD_MIN = LOG_STD_MIN
    self.LOG_STD_MAX = LOG_STD_MAX
    self.use_feature_norm = use_feature_norm

  def __call__(self, x, reset=None, state=None, action_mask=None):
    net, heads = self.build_net()

    x = net(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    
    outs = {}
    for name, layer in heads.items():
      if self.is_action_discrete[name]:
        logits = layer(x)
        if action_mask is not None:
          jax_assert.assert_shape_compatibility([logits, action_mask])
          logits = jnp.where(action_mask, logits, -jnp.inf)
        outs[name] = logits
      else:
        mu = layer[0](x)
        logstd = layer[1](x)
        logstd = jnp.clip(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        scale = lax.exp(logstd)
        outs[name] = (mu, scale)

    return outs, state

  @hk.transparent
  def build_net(self):
    net = MLP(**self.config)
    out_kwargs = net.out_kwargs
    if isinstance(self.action_dim, dict):
      heads = {}
      for k, v in self.action_dim.items():
        if self.is_action_discrete[k]:
          heads[k] = Layer(v, **out_kwargs, name=f'head_{k}')
        else:
          heads[k] = (
            Layer(v, **out_kwargs, name=f'head_{k}_mu'), 
            Layer(v, **out_kwargs, name=f'head_{k}_logstd')
          )
    else:
      raise NotImplementedError(self.action_dim)

    return net, heads


@nn_registry.register('Q')
class Q(hk.Module):
  def __init__(
    self, 
    is_action_discrete, 
    out_size=1, 
    name='q', 
    use_feature_norm=False, 
    **config
  ):
    super().__init__(name=name)
    self.config = dict2AttrDict(config, to_copy=True)
    self.is_action_discrete = is_action_discrete
    self.out_size = out_size
    self.use_feature_norm = use_feature_norm

  def __call__(self, x, a, reset=None, state=None):
    if self.use_feature_norm:
      ln = hk.LayerNorm(-1, True, True)
      x = ln(x)

    x = jnp.concatenate([x, *a.values()], -1)
    net = self.build_net()

    x = net(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    # if self.is_action_discrete:
    #   if a.ndim < x.ndim:
    #     a = nn.one_hot(a, self.out_size)
    #   assert x.ndim == a.ndim, (x.shape, a.shape)
    #   x = jnp.sum(x * a, -1)
    # else:
    #   x = jnp.squeeze(x, -1)
    x = jnp.squeeze(x, -1)

    return x, state

  @hk.transparent
  def build_net(self):
    net = MLP(**self.config, out_size=self.out_size)

    return net


CONSTANT_TEMP = 'constant'
VARIABLE_TEMP = 'variable'


@nn_registry.register('temp')
class Temperature(hk.Module):
  def __init__(
    self, 
    type, 
    value, 
    name='temperature', 
    **config
  ):
    super().__init__(name=name)

    self._type = type
    assert self._type in (VARIABLE_TEMP, CONSTANT_TEMP)
    self._value = value
  
  @property
  def type(self):
    return self._type
  
  @property
  def is_trainable(self):
    return self._type != CONSTANT_TEMP

  def __call__(self):
    if self._type == VARIABLE_TEMP:
      temp_init = hk.initializers.Constant(lax.log(float(self._value)))
      log_temp = hk.get_parameter(
        'log_temp', 
        shape=(), 
        init=temp_init
      )
      temp = lax.exp(log_temp)
      return log_temp, temp
    else:
      temp = jnp.array(self._value, dtype=jnp.float32)
      log_temp = lax.log(temp)
      return log_temp, temp
