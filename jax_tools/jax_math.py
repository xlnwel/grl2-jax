from typing import Dict, Union
from jax import lax, nn, scipy
import jax.numpy as jnp

from . import jax_assert

def safe_ratio(pi, mu, eps=1e-8):
    return pi / (mu + eps)

def upper_clip(x, threshold):
    return x if threshold is None else jnp.minimum(threshold, x)

def lower_clip(x, threshold):
    return x if threshold is None else jnp.maximum(threshold, x)

""" Masked Mathematic Operations """
def _compute_n(mask, n):
    if mask is not None and n is None:
        n = jnp.sum(mask)
        n = jnp.where(n == 0, 1., n)
    return n

def mask_mean(x, mask=None, n=None, axis=None):
    n = _compute_n(mask, n)
    return jnp.mean(x, axis=axis) if mask is None \
        else jnp.sum(x * mask, axis=axis) / n

def mask_moments(x, mask=None, n=None, axis=None):
    n = _compute_n(mask, n)
    mean = mask_mean(x, mask=mask, n=n, axis=axis)
    var = mask_mean((x - mean)**2, mask=mask, n=n, axis=axis)
    return mean, var

def standard_normalization(
    x, 
    zero_center=True, 
    mask=None, 
    n=None, 
    axis=None, 
    epsilon=1e-8, 
):
    mean, var = mask_moments(x, mask=mask, n=n, axis=axis)
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
    ev = mask_mean(ev, mask=mask, n=n)
    return ev

def softmax(x, tau, axis=-1):
    """ sfotmax(x / tau) """
    return nn.softmax(x / tau, axis=axis)

def logsumexp(x, tau, axis=None, keepdims=False):
    """ tau * tf.logsumexp(x / tau) """
    y = scipy.special.logsumexp(x, axis=axis, keepdims=keepdims, b=tau)
    return tau * y
