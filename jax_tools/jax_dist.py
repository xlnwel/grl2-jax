import numpy as np
from jax import lax
import distrax

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
      return {
        'loc': self._loc, 
        'scale': self.scale_diag, 
      }
    else:
      return {
        f'{prefix}_loc': self._loc, 
        f'{prefix}_scale': self.scale_diag, 
      }

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
