import numpy as np
from jax import lax, nn, random
from jax import numpy as jnp
import distrax
from distrax._src.utils import math

EPSILON = 1e-8


# class Distribution:
#   def log_prob(self, x):
#     return -self.neg_log_prob(x)

#   def neg_log_prob(self, x):
#     raise NotImplementedError

#   def sample(self, rng, *args, **kwargs):
#     raise NotImplementedError
    
#   def entropy(self):
#     raise NotImplementedError

#   def kl(self, other):
#     assert isinstance(other, type(self))
#     raise NotImplementedError

#   def mean(self):
#     raise NotImplementedError

#   def mode(self):
#     raise NotImplementedError


# class Categorical(Distribution):
#   """ An implementation of tfd.RelaxedOneHotCategorical """
#   def __init__(self, logits, tau=None, epsilon=EPSILON):
#     self.logits = logits
#     self.tau = tau  # tau in Gumbel-Softmax
#     self.epsilon = epsilon

#   def neg_log_prob(self, x, mask=None):
#     logits = self.get_masked_logits(mask)
#     if x.shape == self.logits.shape:
#       # when x is one-hot encoded
#       return optax.softmax_cross_entropy(logits, x)
#     else:
#       return optax.softmax_cross_entropy_with_integer_labels(logits, x)

#   def sample(self, rng, mask=None, one_hot=False):
#     """
#      A differentiable sampling method for categorical distribution
#      reference paper: Categorical Reparameterization with Gumbel-Softmax
#      original code: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
#     """
#     logits = self.get_masked_logits(mask)
#     if self.tau and one_hot:
#       # sample Gumbel(0, 1)
#       U = random.uniform(rng, shape=logits.shape, 
#         dtype=logits.dtype, minval=0, maxval=1)
#       g = -lax.log(-lax.log(U+self.epsilon)+self.epsilon)
#       # Draw a sample from the Gumbel-Softmax distribution
#       y = nn.softmax((logits + g) / self.tau)
#       # draw one-hot encoded sample from the softmax
#     else:
#       y = random.categorical(rng, logits=logits)
#       if one_hot:
#         y = nn.one_hot(y, logits.shape[-1], dtype=logits.dtype)

#     return y

#   def entropy(self, mask=None):
#     logits = self.get_masked_logits(mask)
#     probs = nn.softmax(logits)
#     log_probs = nn.log_softmax(logits)
#     entropy = -jnp.sum(probs * log_probs, axis=-1)

#     return entropy

#   def kl(self, other: Distribution, mask=None):
#     logits = self.get_masked_logits(mask)
#     other_logits = other.get_masked_logits(mask)
#     probs = nn.softmax(logits)
#     log_probs = nn.log_softmax(logits)
#     other_log_probs = nn.log_softmax(other_logits)
#     log_ratio = log_probs - other_log_probs
#     kl = jnp.sum(probs * log_ratio, axis=-1)

#     return kl

#   def mode(self, one_hot=False):
#     y = jnp.argmax(self.logits, -1)
#     if one_hot:
#       y = nn.one_hot(y, self.logits.shape[-1])
#     return y
  
#   def get_masked_logits(self, mask):
#     if mask is not None:
#       logits = jnp.where(mask, self.logits, -1e10)
#     else:
#       logits = self.logits
#     return logits

#   def stop_gradient(self):
#     self.logits = lax.stop_gradient(self.logits)

#   def get_stats(self, prefix=None):
#     if prefix is None:
#       return {'logits': self.logits}
#     else:
#       return {f'{prefix}_logits': self.logits}


# class MultivariateNormalDiag(Distribution):
#   def __init__(self, mean, logstd, epsilon=EPSILON):
#     self.mu, self.logstd = mean, logstd
#     self.std = lax.exp(self.logstd)
#     self.epsilon = epsilon

#   def neg_log_prob(self, x):
#     return .5 * jnp.sum(lax.log(
#       2. * np.pi)
#       + 2. * self.logstd
#       + ((x - self.mu) / (self.std + self.epsilon))**2, 
#     axis=-1)

#   def sample(self, rng):
#     return self.mu + self.std * random.normal(
#       rng, self.mu.shape, dtype=self.mu.dtype)

#   def entropy(self):
#     return jnp.sum(.5 * np.log(2. * np.pi) + self.logstd + .5, axis=-1)

#   def kl(self, other):
#     return jnp.sum(other.logstd - self.logstd - .5
#             + .5 * (self.std**2 + (self.mu - other.mean)**2)
#             / (other.std + self.epsilon)**2, axis=-1)

#   def mean(self):
#     return self.mu
  
#   def mode(self):
#     return self.mu

#   def stop_gradient(self):
#     self.mu = lax.stop_gradient(self.mu)
#     self.logstd = lax.stop_gradient(self.logstd)
#     self.std = lax.stop_gradient(self.std)

#   def get_stats(self, prefix=None):
#     if prefix is None:
#       return {
#         'mean': self.mu, 
#         'std': self.std, 
#       }
#     else:
#       return {
#         f'{prefix}_mean': self.mu, 
#         f'{prefix}_std': self.std, 
#       }


class Bernoulli(distrax.Bernoulli):
  def stop_gradient(self):
    logits = None if self._logits is None else lax.stop_gradient(self._logits)
    probs = None if self._probs is None else lax.stop_gradient(self._probs)
    super().__init__(logits=logits, probs=probs, dtype=self._dtype)

  def get_stats(self, prefix=None):
    if prefix is None:
      return {'logits': self._logits}
    else:
      return {f'{prefix}_logits': self._logits}


class Categorical(distrax.Categorical):
  def stop_gradient(self):
    logits = None if self._logits is None else lax.stop_gradient(self._logits)
    probs = None if self._probs is None else lax.stop_gradient(self._probs)
    super().__init__(logits=logits, probs=probs, dtype=self._dtype)

  def get_stats(self, prefix=None):
    if prefix is None:
      return {'logits': self._logits}
    else:
      return {f'{prefix}_logits': self._logits}
  
  @staticmethod
  def stats_keys(prefix=None):
    if prefix is None:
      return ('logits',)
    else:
      return (f'{prefix}_logits',)


class MultiCategorical(distrax.Distribution):
  def __init__(self, logits, split_indices, dtype=int):
    logits = jnp.split(logits, split_indices, axis=-1)
    self._logits = [math.normalize(logits=l) for l in logits]
    self._probs = None
    self._dtype = dtype
  
  @property
  def logits(self):
    return self._logits
  
  @property
  def probs(self):
    if self._probs is None:
      self._probs = [nn.softmax(logits) for logits in self._logits]
    return self._probs

  @property
  def event_shape(self):
    return (len(self._logits),)

  @property
  def num_categories(self):
    return [logits.shape[-1] for logits in self._logits]

  def _sample_n(self, rng, n: int):
    """See `Distribution._sample_n`."""
    new_shapes = [(n,) + l.shape[:-1] for l in self.logits]
    is_valid = [jnp.logical_and(
      jnp.all(jnp.isfinite(p), axis=-1),
      jnp.all(p >= 0, axis=-1)
    ) for p in self.probs]
    rngs = random.split(rng, len(self.logits))
    draws = [random.categorical(
      key=k, logits=l, axis=-1,
      shape=s).astype(self._dtype)
      for k, l, s in zip(rngs, self.logits, new_shapes)]
    return jnp.stack(
      [jnp.where(v, d, jnp.ones_like(d) * -1) for v, d in zip(is_valid, draws)
    ], axis=-1)

  def stop_gradient(self):
    self._logits = lax.stop_gradient(self._logits)

  def log_prob(self, values):
    values = jnp.split(values, values.shape[-1], axis=-1)
    values = [jnp.squeeze(v, -1) for v in values]
    assert len(self._logits) == len(values), (len(self._logits), len(values))
    values_oh = [nn.one_hot(v, n, dtype=l.dtype) 
                 for v, n, l in zip(values, self.num_categories, self.logits)]
    masks_outside_domain = [jnp.logical_or(v < 0, v > n - 1) 
                           for v, n in zip(values, self.num_categories)]
    lps = [jnp.where(m, -jnp.inf, jnp.sum(math.multiply_no_nan(l, v), axis=-1)) 
           for m, l, v in zip(masks_outside_domain, self.logits, values_oh)]
    return sum(lps)

  def prob(self, values):
    lp = self.log_prob(values)
    return lax.exp(lp)

  def entropy(self):
    log_probs = [lax.log(p) for p in self.probs]
    ents = [-jnp.sum(math.mul_exp(lp, lp), axis=-1) for lp in log_probs]
    return sum(ents)
  
  def mode(self):
    v = jnp.stack([
      jnp.argmax(l, axis=-1).astype(self._dtype) for l in self._logits
    ], axis=-1)
    return v
  
  def cdf(self, values):
    should_be_zero = [v < 0 for v in values]
    should_be_one = [v >= n for v, n in zip(values, self.num_categories)]
    values = [jnp.clip(v, 0, n-1) for v, n in zip(values, self.num_categories)]
    values_one_hot = [nn.one_hot(v, n, dtype=self.logits.dtype) 
                      for v, n in zip(values, self.num_categories)]
    cdfs = [jnp.sum(math.multiply_no_nan(
      jnp.cumsum(p, axis=-1), v), axis=-1
    ) for p, v in zip(self.probs, values_one_hot)]
    return [jnp.where(sbz, 0., jnp.where(sbo, 1., cdf)) 
            for sbz, sbo, cdf in zip(should_be_zero, should_be_one, cdfs)]

  def get_stats(self, prefix=None):
    logits = jnp.concatenate(self._logits, -1)
    if prefix is None:
      stats = {'logits': logits}
    else:
      stats = {f'{prefix}_logits': logits}
    return stats
  
  @staticmethod
  def stats_keys(prefix=None):
    if prefix is None:
      return ('logits',)
    else:
      return (f'{prefix}_logits',)

class MultivariateNormalDiag(distrax.MultivariateNormalDiag):
  def __init__(self, loc, scale=None, joint_log_prob=True):
    super().__init__(loc, scale)
    self._joint_log_prob = joint_log_prob

  # @property
  # def scale(self):
  #   return self.scale_diag

  def stop_gradient(self):
    loc = lax.stop_gradient(self.loc)
    scale = lax.stop_gradient(self._scale_diag)
    super().__init__(loc, scale)

  def get_stats(self, prefix=None):
    if prefix is None:
      stats= {
        'loc': self._loc, 
        'scale': self.scale_diag, 
      }
      stats.update({
        f'loc{i}': self._loc[..., i] for i in range(self._loc.shape[-1])
      })
      stats.update({
        f'scale{i}': self._scale_diag[..., i] for i in range(self._scale_diag.shape[-1])
      })
    else:
      stats= {
        f'{prefix}_loc': self._loc, 
        f'{prefix}_scale': self.scale_diag, 
      }
      stats.update({
        f'{prefix}_loc{i}': self._loc[..., i] for i in range(self._loc.shape[-1])
      })
      stats.update({
        f'{prefix}_scale{i}': self._scale_diag[..., i] for i in range(self._scale_diag.shape[-1])
      })
    return stats

  @staticmethod
  def stats_keys(prefix=None):
    if prefix is None:
      return ('loc', 'scale')
    else:
      return (f'{prefix}_loc', f'{prefix}_scale')

  def sample_and_log_prob(self, *, seed, joint=None):
    if joint is None:
      joint = self._joint_log_prob
    if joint:
      return super().sample_and_log_prob(seed=seed)
    else:
      action = super().sample(seed=seed)
      logprob = self._ind_log_prob(action)
      return action, logprob

  def log_prob(self, x, joint=None):
    if joint is None:
      joint = self._joint_log_prob
    if joint:
      return super().log_prob(x)
    else:
      return self._ind_log_prob(x)

  def _ind_log_prob(self, x):
    return -lax.log(self.scale_diag) -.5 * (
      lax.log(2. * np.pi) + ((x - self._loc) / self.scale_diag)**2)
