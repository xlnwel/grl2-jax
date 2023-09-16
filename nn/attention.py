import haiku as hk
import jax
from jax import numpy as jnp

from core.typing import dict2AttrDict
from nn.mlp import MLP
from nn.registry import nn_registry
from nn.utils import dropout


CAUSAL_MASK = 'causal'


@nn_registry.register('attn')
class Attention(hk.Module):
  def __init__(
    self, 
    name='attention', 
    mask=None, 
    scale_logits=True, 
    n_heads=1, 
    attn_dropout=0, 
    res_dropout=0, 
    **config
  ):
    super().__init__(name=name)
    self.config = dict2AttrDict(config)
    self.mask = mask
    self.scale_logits = scale_logits
    self.n_heads = n_heads
    self.attn_dropout = attn_dropout
    self.res_dropout = res_dropout
    assert self.mask in [None, CAUSAL_MASK], self.mask

  def __call__(self, x, kv=None, mask=None, training=False):
    if kv is None:
      kv = x
    ql, kl, vl, pl = self.build_net(x)
    
    q = ql(x)
    k = kl(kv)
    v = vl(kv)

    if self.n_heads > 1:
      q, k, v = jax.tree_util.tree_map(
        lambda x: _reshape_for_mhsa(x, self.n_heads), [q, k, v])

    mask = self._get_mask(q.shape[-2], mask=mask)
    x = attention(
      q, k, v, 
      mask=mask, 
      scale_logits=self.scale_logits, 
      attn_dropout=self.attn_dropout, 
      training=training
    )

    if self.n_heads > 1:
      x = _recover_shape(x)

    x = pl(x)
    x = dropout(self.res_dropout, training, x)

    return x

  @hk.transparent
  def build_net(self, x):
    if self.config.out_size is None:
      self.config.out_size = x.shape[-1]
    assert self.config.out_size % self.n_heads == 0, (self.config.out_size, self.n_heads)
    query_layer = MLP(**self.config, name='query')
    key_layer = MLP(**self.config, name='key')
    value_layer = MLP(**self.config, name='value')
    proj_layer = MLP(**self.config, name='projection')

    return query_layer, key_layer, value_layer, proj_layer

  def _get_mask(self, T, mask=None):
    if mask is None:
      mask = self.mask
    if mask == CAUSAL_MASK:
      mask = jnp.tri(T)
    
    return mask


def _reshape_for_mhsa(x, n_heads):
  x = x.reshape(*x.shape[:-1], n_heads, x.shape[-1] // n_heads)
  x = jnp.swapaxes(x, -2, -3)
  return x


def _recover_shape(x):
  x = jnp.swapaxes(x, -2, -3)
  x = x.reshape(*x.shape[:-2], -1)
  return x


def attention(
  q, k, v, 
  mask=None, 
  scale_logits=False, 
  attn_dropout=0, 
  training=False, 
):
  """ compute softmax(qk^T)v """
  if scale_logits:
    q *= k.shape[-1] ** -.5
  dot_product = q @ jnp.swapaxes(k, -2, -1)
  if mask is not None:
    dot_product = jnp.where(mask, dot_product, float('-inf'))
  weights = jax.nn.softmax(dot_product)
  weights = dropout(attn_dropout, training, weights)
  x = weights @ v

  return x
