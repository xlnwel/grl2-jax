import collections
import numpy as np

from core.log import do_logging
from tools.utils import expand_dims_match, moments


StatsWithVar = collections.namedtuple('RMS', 'mean var')
StatsWithVarCount = collections.namedtuple('RMS', 'mean var count')
StatsWithStd = collections.namedtuple('RMS', 'mean std')
StatsWithStdCount = collections.namedtuple('RMS', 'mean std count')


def combine_rms(rms1, rms2):
  # if np.any(np.abs(rms1.mean - rms2.mean) > 100) \
  #   or np.any(np.abs(rms1.var - rms2.var) > 100):
  #   do_logging(f'Large difference between two RMSs {rms1} vs {rms2}')

  mean1, var1, count1 = rms1
  mean2, var2, count2 = rms2
  delta = mean2 - mean1
  total_count = count1 + count2
  if total_count == 0:
    return rms1

  new_mean = mean1 + delta * count2 / total_count
  # no minus one here to be consistent with np.std
  m_a = var1 * count1
  m_b = var2 * count2
  M2 = m_a + m_b + delta**2 * count1 * count2 / total_count
  # if not np.all(np.isfinite(M2)):
  #   breakpoint()
  assert np.all(np.isfinite(M2)), f'M2: {M2}'
  new_var = M2 / total_count

  return StatsWithVarCount(new_mean, new_var, total_count)


def denormalize(x, mean, std, zero_center=True, mask=None, 
        dim_mask=None, np=np):
  """ Denormalize x using mean and std
  mask chooses which samples to apply denormalization
  dim_mask masks out dimensions with small variance
  """
  std = std if dim_mask is None else np.where(dim_mask, std, 1.)
  x_new = x * std
  if zero_center:
    x_new = x_new + mean
  if mask is not None:
    mask = expand_dims_match(mask, x_new)
    x_new = np.where(mask, x_new, x)
  return x_new


def normalize(x, mean, std, zero_center=True, clip=None, mask=None, 
        dim_mask=None, np=np):
  """ Normalize x using mean and std
  mask chooses which samples to apply normalization
  dim_mask masks out dimensions with small variance
  """
  x_new = x
  if zero_center:
    x_new = x_new - mean
  std = std if dim_mask is None else np.where(dim_mask, std, 1.)
  x_new = x_new / std
  if clip:
    x_new = np.clip(x_new, -clip, clip)
  if mask is not None:
    mask = expand_dims_match(mask, x_new)
    x_new = np.where(mask, x_new, x)
  return x_new


class RunningMeanStd:
  # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  def __init__(self, axis, epsilon=1e-8, clip=None, name=None, ndim=None, const_threshold=1e-3):
    """ Computes running mean and std from data
    A reimplementation of RunningMeanStd from OpenAI's baselines

    Args:
      axis: axis along which we compute mean and std from incoming data. 
        If it's None, we only receive at a time a sample without batch dimension
      ndim: expected number of dimensions for the stats, useful for debugging
    """
    self.name = name
    # assertion to avoid any unexpected construction
    assert name is not None, name

    if isinstance(axis, int):
      axis = (axis, )
    elif isinstance(axis, (tuple, list)):
      axis = tuple(axis)
    elif axis is None:
      pass
    else:
      raise ValueError(f'Invalid axis({axis}) of type({type(axis)})')

    if isinstance(axis, tuple):
      assert axis == tuple(range(len(axis))), \
        f'Axis should only specifies leading axes so that '\
        f'mean and var can be broadcasted automatically when normalizing. '\
        f'But receving axis = {axis}'
    self._axis = axis
    if self._axis is not None:
      self._shape_slice = np.s_[: max(self._axis)+1]
    self._epsilon = epsilon
    self._clip = clip
    self._ndim = ndim # expected number of dimensions
    self._const_threshold = const_threshold
    assert self._const_threshold > np.sqrt(self._epsilon), (self._const_threshold, self._epsilon)
    self.reset_rms_stats()

  @property
  def axis(self):
    return self._axis

  @property
  def mean(self):
    return self._mean

  @property
  def var(self):
    return self._var

  @property
  def std(self):
    return self._std

  @property
  def count(self):
    return self._count

  def const_dim_mask(self):
    dim_mask = self._std > self._const_threshold
    return dim_mask

  def is_initialized(self):
    return self._count > 0

  def reset_rms_stats(self):
    if self._ndim:
      shape = [1 for _ in range(self._ndim)]
      self._mean = np.zeros(shape)
      self._var = np.ones(shape)
      self._std = np.ones(shape)
    else:
      self._mean = 0
      self._var = 1
      self._std = 1

    self._count = 0

  def set_rms_stats(self, mean, var, count):
    self._mean = mean
    self._var = var
    self._std = np.sqrt(self._var + self._epsilon)
    self._count = count

  def get_rms_stats(self, with_count=True, return_std=False):
    if with_count:
      if return_std:
        return StatsWithStdCount(self._mean, self._std, self._count)
      else:
        return StatsWithVarCount(self._mean, self._var, self._count)
    else:
      if return_std:
        return StatsWithStd(self._mean, self._std)
      else:
        return StatsWithVar(self._mean, self._var)

  def update(self, x, mask=None, axis=None):
    x = x.astype(np.float64)
    if axis is None:
      axis = self._axis
      shape_slice = self._shape_slice
    else:
      if isinstance(axis, int):
        axis = (axis,)
      shape_slice = np.s_[:max(axis)+1]
    if axis is None:
      assert mask is None, mask
      batch_mean, batch_var, batch_count = x, np.zeros_like(x), 1
    else:
      batch_mean, batch_var = moments(x, axis, mask)
      batch_count = np.prod(x.shape[shape_slice]) \
        if mask is None else np.sum(mask)
    if batch_count > 0:
      if self._ndim is not None:
        assert batch_mean.ndim == self._ndim, (batch_mean.shape, self._ndim)
      self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(self, batch_mean, batch_var, batch_count):
    if batch_count == 0:
      return
    assert np.all(batch_var >= 0), batch_var[batch_var < 0]
    if self._count == 0:
      self._mean = np.zeros_like(batch_mean, 'float64')
      self._var = np.ones_like(batch_var, 'float64')
    assert self._mean.shape == batch_mean.shape
    assert self._var.shape == batch_var.shape

    new_mean, new_var, total_count = combine_rms(
      StatsWithVarCount(self._mean, self._var, self._count), 
      StatsWithVarCount(batch_mean, batch_var, batch_count), 
    )
    self._mean = new_mean
    self._var = new_var
    self._std = np.sqrt(self._var + self._epsilon)
    self._count = total_count

  def normalize(self, x, zero_center=True, mask=None, ignore_const=True):
    assert not np.isinf(np.std(x)), f'{np.min(x)}\t{np.max(x)}'
    assert self._std is not None, (self._mean, self._std, self._count)
    if self._count == self._epsilon:
      if self._clip:
        x = np.clip(x, -self._clip, self._clip)
      return x
    # assert x.ndim == self._var.ndim + (0 if self._axis is None else len(self._axis)), \
    #   (x.shape, self._var.shape, self._axis)
    dim_mask = self.const_dim_mask() if ignore_const else None
    x_new = x.astype(np.float32)
    x_new = normalize(
      x_new, self._mean, self._std,
      zero_center=zero_center, 
      clip=self._clip, mask=mask, 
      dim_mask=dim_mask
    )
    return x_new

  def denormalize(self, x, zero_center=True, mask=None, ignore_const=True):
    assert not np.isinf(np.std(x)), f'{np.min(x)}\t{np.max(x)}'
    assert self._std is not None, (self._mean, self._std, self._count)
    # assert x.ndim == self._var.ndim + (0 if self._axis is None else len(self._axis)), \
    #   (x.shape, self._var.shape, self._axis)
    if self._count == self._epsilon:
      return x
    dim_mask = self.const_dim_mask() if ignore_const else None
    x_new = x.astype(np.float32)
    x_new = denormalize(
      x_new, self._mean, self._std, 
      zero_center=zero_center, mask=mask, 
      dim_mask=dim_mask, 
    )
    return x_new


# class TFRunningMeanStd:
#   """ Different from PopArt, this is only for on-policy training, """
#   def __init__(self, axis, shape=(), clip=None, epsilon=1e-2, dtype=tf.float32):
#     # use tf.float64 to avoid overflow
#     self._sum = tf.Variable(np.zeros(shape), trainable=False, dtype=tf.float64, name='sum')
#     self._sumsq = tf.Variable(np.zeros(shape), trainable=False, dtype=tf.float64, name='sum_squares')
#     self._count = tf.Variable(np.zeros(shape), trainable=False, dtype=tf.float64, name='count')
#     self._mean = None
#     self._std = None
#     self._axis = axis
#     self._clip = clip
#     self._epsilon = epsilon
#     self._dtype = dtype

#   def update(self, x):
#     x = tf.cast(x, tf.float64)
#     self._sum.assign_add(tf.reduce_sum(x, axis=self._axis))
#     self._sumsq.assign_add(tf.cast(tf.reduce_sum(x**2, axis=self._axis), self._sumsq.dtype))
#     self._count.assign_add(tf.cast(tf.math.reduce_prod(tf.shape(x)[:len(self._axis)]), self._count.dtype))
#     mean = self._sum / self._count
#     std = tf.sqrt(tf.maximum(
#       self._sumsq / self._count - mean**2, self._epsilon))
#     self._mean = tf.cast(mean, self._dtype)
#     self._std = tf.cast(std, self._dtype)

#   def normalize(self, x, zero_center=True):
#     if zero_center:
#       x = x - self._mean
#     x = x / self._std
#     if self._clip is not None:
#       x = tf.clip_by_value(x, -self._clip, self._clip)
#     return x
  
#   def denormalize(self, x, zero_center=True):
#     x = x * self._std
#     if zero_center:
#       x = x + self._mean
#     return x
