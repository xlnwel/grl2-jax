from typing import Dict, Union
from jax import lax, nn, scipy
import jax.numpy as jnp

from . import jax_assert


def safe_ratio(pi, mu, eps=1e-8):
  return pi / (mu + eps)

def center_clip(x, threshold):
  return x if threshold is None else jnp.clip(x, 1-threshold, 1+threshold)

def upper_clip(x, threshold):
  return x if threshold is None else jnp.minimum(threshold, x)

def lower_clip(x, threshold):
  return x if threshold is None else jnp.maximum(threshold, x)

""" Masked Mathematic Operations """
def count_masks(mask, axis=None, n=None):
  if mask is not None and n is None:
    n = jnp.sum(mask, axis=axis)
    n = jnp.where(n == 0, 1., n)
  return n

def mask_mean(x, mask=None, replace=0, axis=None):
  if mask is None:
    x = jnp.mean(x, axis=axis)
  elif replace is None:
    n = count_masks(mask, axis=axis)
    x = jnp.sum(x * mask, axis=axis) / n
  else:
    x = jnp.where(mask, x, replace)
    x = jnp.mean(x, axis=axis)
  return x

def mask_moments(x, mask=None, axis=None):
  mean = mask_mean(x, mask=mask, replace=None, axis=axis)
  var = mask_mean((x - mean)**2, mask=mask, replace=None, axis=axis)
  return mean, var

def standard_normalization(
  x, 
  zero_center=True, 
  mask=None, 
  axis=None, 
  epsilon=1e-8, 
):
  mean, var = mask_moments(x, mask=mask, axis=axis)
  std = lax.sqrt(var + epsilon)
  if zero_center:
    x = x - mean
  x = x / std

  return x

def clip(x, clip: Union[int, float, Dict]):
  if clip:
    if isinstance(clip, dict):
      pos = clip['pos']
      neg = clip['neg']
    else:
      pos = clip
      neg = -clip
    x = jnp.clip(x, neg, pos)

  return x

def explained_variance(y, pred, axis=None, mask=None, n=None):
  jax_assert.assert_shape_compatibility([y, pred])
  y_var = jnp.var(y, axis=axis) + 1e-8
  diff_var = jnp.var(y - pred, axis=axis)
  ev = jnp.maximum(-1., 1-(diff_var / y_var))
  ev = mask_mean(ev, mask=mask, replace=None)
  return ev

def softmax(x, tau, axis=-1):
  """ sfotmax(x / tau) """
  return nn.softmax(x / tau, axis=axis)

def logsumexp(x, tau, axis=None, keepdims=False):
  """ tau * tf.logsumexp(x / tau) """
  y = scipy.special.logsumexp(x / tau, axis=axis, keepdims=keepdims)
  return tau * y

def symlog(x):
  return lax.sign(x) * lax.log(lax.abs(x) + 1)

def symexp(x):
  return lax.sign(x) * (lax.exp(lax.abs(x)) - 1)
