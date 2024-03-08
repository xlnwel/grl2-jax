import numpy as np
import jax
from jax import random
import jax.numpy as jnp


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
sg = lambda x: jax.tree_map(jax.lax.stop_gradient, x)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=jnp.float32):
    super().__init__(logits, probs, dtype)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
     return super()._parameter_properties(dtype)

  def sample(self, sample_shape=(), seed=None):
    sample = sg(super().sample(sample_shape, seed))
    probs = self._pad(super().probs_parameter(), sample.shape)
    return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

  def _pad(self, tensor, shape):
    while len(tensor.shape) < len(shape):
      tensor = tensor[None]
    return tensor


class Dist:

  def __init__(
      self, shape, outscale=0.1, outnorm=False, minstd=1.0,
      maxstd=1.0, unimix=0.0, bins=255, val_min=-20, val_max=20):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale
    self._outnorm = outnorm
    self._bins = bins
    self._val_min = val_min
    self._val_max = val_max

  def __call__(self, inputs, aux_inp=None):
    dist = self.inner(inputs, aux_inp)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs, aux_inp=None):
    kw = {}
    kw['outscale'] = self._outscale
    kw['outnorm'] = self._outnorm
    shape = self._shape
    out = inputs.reshape(inputs.shape[:-1] + shape).astype(jnp.float32)
    if aux_inp is not None:
      mask = aux_inp['mask'].astype(bool)
      out = jnp.where(mask, out, -jnp.inf)
    if self._unimix:
      probs = jax.nn.softmax(out, -1)
      if aux_inp is None:
        uniform = jnp.ones_like(probs) / probs.shape[-1]
      else:
        mask_ones = jnp.where(mask, 1, 0)
        uniform = mask_ones / jnp.sum(mask, axis=-1, keepdims=True)
      probs = (1 - self._unimix) * probs + self._unimix * uniform
      out = jnp.log(probs)
    dist = OneHotDist(out)
    if len(self._shape) > 1:
      dist = tfd.Independent(dist, len(self._shape) - 1)
    dist.minent = 0.0
    dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
    return dist


if __name__ == '__main__':
  rng = random.PRNGKey(42)
  rngs = random.split(rng, 2)
  dim = 5
  shape = (10, 5)
  mask = random.randint(rngs[0], shape, 0, 2)
  x = random.normal(rngs[1], shape)
  d = Dist((5,))(x, {'mask': mask})
  sample = d.sample(seed=rng)
  logp = d.log_prob(sample)
  print('logp', logp)