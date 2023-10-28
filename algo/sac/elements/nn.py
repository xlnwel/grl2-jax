from jax import lax, nn
import jax.numpy as jnp
import haiku as hk

from core.typing import dict2AttrDict
from nn.func import nn_registry
from nn.mlp import MLP
from jax_tools import jax_assert
from algo.masac.elements.utils import concat_sa


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
    net = self.build_net()

    x = net(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    
    if self.is_action_discrete:
      if action_mask is not None:
        jax_assert.assert_shape_compatibility([x, action_mask])
        x = jnp.where(action_mask, x, -1e10)
      return x, state
    else:
      mu, logstd = jnp.split(x, 2, -1)
      logstd = jnp.clip(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
      scale = lax.exp(logstd)
      return (mu, scale), state

  @hk.transparent
  def build_net(self):
    out_size = self.action_dim if self.is_action_discrete else 2 * self.action_dim
    net = MLP(
      **self.config, 
      out_size=out_size, 
    )

    return net


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

    if not self.is_action_discrete:
      x = concat_sa(x, a)
    net = self.build_net()

    x = net(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    if self.is_action_discrete:
      if a.ndim < x.ndim:
        a = nn.one_hot(a, self.out_size)
      assert x.ndim == a.ndim, (x.shape, a.shape)
      x = jnp.sum(x * a, -1)
    else:
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
    temp_init = hk.initializers.Constant(lax.log(self._value))
    temp = jnp.array(self._value)
    if self._type == VARIABLE_TEMP:
      log_temp = hk.get_parameter(
        'log_temp', 
        shape=(), 
        init=temp_init
      )
      temp = lax.exp(log_temp)
      return log_temp, temp
    else:
      log_temp = lax.log(temp)
      return log_temp, temp
