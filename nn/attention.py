import haiku as hk
import jax
from jax import numpy as jnp

from core.typing import dict2AttrDict
from nn.registry import nn_registry
from nn.utils import dropout


CAUSAL_MASK = 'causal'


@nn_registry.register('attn')
class Attention(hk.Module):
  def __init__(
    self, 
    name='attention', 
    scale_logits=True, 
    n_heads=1, 
    attn_dropout=0, 
    res_dropout=0, 
    **config
  ):
    super().__init__(name=name)
    self.config = dict2AttrDict(config)
    self.scale_logits = scale_logits
    self.n_heads = n_heads
    self.attn_dropout = attn_dropout
    self.res_dropout = res_dropout
    assert self.mask in [None, CAUSAL_MASK], self.mask

  def __call__(self, x, kv=None, mask=None, 
               data_mask=None, query_mask=None, 
               kv_mask=None, training=False):
    if kv is None:
      kv = x
    ql, kl, vl, pl = self.build_net(x)
    
    q = ql(x)
    k = kl(kv)
    v = vl(kv)

    if self.n_heads > 1:
      q, k, v = jax.tree_util.tree_map(
        lambda x: _reshape_for_mha(x, self.n_heads), [q, k, v])

    mask = self._get_mask(q.shape[-2], mask=mask)
    mask = get_attention_mask(
      mask, data_mask, 
      query_mask=query_mask, 
      kv_mask=kv_mask, 
      nq=q.shape[-2], 
      nk=k.shape[-2]
    )
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
    dm = get_linear_mask(data_mask, query_mask)
    if dm is not None:
      x = x * dm
    x = dropout(x, self.res_dropout, training)

    return x

  @hk.transparent
  def build_net(self, x):
    if self.config.out_size is None:
      self.config.out_size = x.shape[-1]
    assert self.config.out_size % self.n_heads == 0, (self.config.out_size, self.n_heads)
    query_layer = hk.Linear(**self.config, name='query')
    key_layer = hk.Linear(**self.config, name='key')
    value_layer = hk.Linear(**self.config, name='value')
    proj_layer = hk.Linear(**self.config, name='projection')

    return query_layer, key_layer, value_layer, proj_layer

  def _get_mask(self, T, mask=None):
    if mask == CAUSAL_MASK:
      mask = jnp.tri(T)
    
    return mask


def _reshape_for_mha(x, n_heads):
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
  scale_logits=True, 
  attn_dropout=0, 
  training=False, 
):
  """ compute softmax(qk^T)v """
  if scale_logits:
    q *= k.shape[-1] ** -.5
  dot_product = q @ jnp.swapaxes(k, -2, -1)
  if mask is not None:
    assert mask.shape[-2:] == dot_product.shape[-2:], (mask, dot_product)
    dot_product = jnp.where(mask, dot_product, -jnp.inf)
  weights = jax.nn.softmax(dot_product, axis=-1)
  weights = jnp.nan_to_num(weights)
  weights = dropout(weights, attn_dropout, training)
  x = weights @ v

  return x


def get_attention_mask(mask, data_mask, query_mask, kv_mask, nq, nk):
  if data_mask is not None:
    _data_mask = jnp.expand_dims(data_mask, axis=-1)
    _data_mask = _data_mask @ jnp.swapaxes(_data_mask, -2, -1)
    if mask is None:
      mask = _data_mask
    else:
      mask = jnp.logical_and(_data_mask, mask)
  if query_mask is not None:
    _query_mask = jnp.expand_dims(query_mask, axis=-1)
    _kv_mask = jnp.ones(_query_mask.shape[:-2] + (1, nk))
    _query_mask = _query_mask @ _kv_mask
    if mask is None:
      mask = _query_mask
    else:
      mask = jnp.logical_and(_query_mask, mask)
  if kv_mask is not None:
    _kv_mask = jnp.expand_dims(kv_mask, axis=-2)
    _query_mask = jnp.ones(_kv_mask.shape[:-2] + (nq, 1))
    _kv_mask = _query_mask @ _kv_mask
    if mask is None:
      mask = _kv_mask
    else:
      mask = jnp.logical_and(_kv_mask, mask)
  return mask


def get_linear_mask(data_mask, query_mask):
  if query_mask is not None:
    if data_mask is None:
      data_mask = query_mask
    else:
      data_mask = jnp.logical_and(data_mask, query_mask)
  if data_mask is not None:
    data_mask = jnp.expand_dims(data_mask, -1)
  return data_mask
