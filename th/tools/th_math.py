from typing import Dict, Union
import torch


def safe_ratio(pi, mu, eps=1e-8):
  return pi / (mu + eps)

def center_clip(x, threshold):
  return x if threshold is None else x.clamp(1-threshold, 1+threshold)

def upper_clip(x, threshold):
  return x if threshold is None else x.clamp(max=threshold)

def lower_clip(x, threshold):
  return x if threshold is None else x.clamp(min=threshold)

""" Masked Mathematic Operations """
def count_masks(mask, axis=None, n=None):
  if mask is not None and n is None:
    n = mask.sum(axis)
    n = torch.where(n == 0, 1., n)
  return n

def mask_mean(x, mask=None, replace=0, axis=None):
  if mask is None:
    x = x.mean(axis)
  elif replace is None:
    n = count_masks(mask, axis=axis)
    x = (x * mask).sum(axis) / n
  else:
    x = torch.where(mask.to(bool), x, replace)
    x = x.mean(axis)
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
  std = torch.sqrt(var + epsilon)
  if zero_center:
    x = x - mean
  x = x / std
  x = torch.where(mask.to(bool), x, 0)

  return x

def clip(x, clip: Union[int, float, Dict]):
  if clip:
    if isinstance(clip, dict):
      pos = clip['pos']
      neg = clip['neg']
    else:
      pos = clip
      neg = -clip
    x = torch.clamp(x, neg, pos)

  return x

def explained_variance(y, pred, axis=None, mask=None):
  y_var = torch.var(y, axis) + 1e-8
  diff_var = torch.var(y - pred, axis)
  ev = torch.maximum(torch.tensor(-1), 1-(diff_var / y_var))
  ev = mask_mean(ev, mask=mask, replace=None)
  return ev

def softmax(x, tau, axis=-1):
  """ sfotmax(x / tau) """
  return torch.softmax(x / tau, axis)

def logsumexp(x, tau, axis=None, keepdims=False):
  """ tau * tf.logsumexp(x / tau) """
  y = torch.logsumexp(x / tau, axis, keepdims=keepdims)
  return tau * y

def symlog(x):
  return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
  return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
