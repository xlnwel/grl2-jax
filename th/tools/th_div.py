from jax import lax
import jax.numpy as jnp
import chex
import distrax


def kl_divergence(
  *, 
  reg_type,
  logp=None,
  logq=None, 
  sample_prob=1., 
  p_logits=None,
  q_logits=None,
  p_loc=None,
  p_scale=None,
  q_loc=None,
  q_scale=None,
  logits_mask=None, 
):
  if reg_type == 'forward_approx':
    kl = kl_from_samples(
      logp=logq, 
      logq=logp, 
      sample_prob=sample_prob, 
    )
  elif reg_type == 'reverse_approx':
    kl = reverse_kl_from_samples(
      logp=logp, 
      logq=logq, 
      sample_prob=sample_prob, 
    )
  elif reg_type == 'forward':
    kl = kl_from_distributions(
      p_logits=q_logits, 
      q_logits=p_logits, 
      p_loc=q_loc, 
      p_scale=q_scale, 
      q_loc=p_loc, 
      q_scale=p_scale, 
      logits_mask=logits_mask, 
    )
  elif reg_type == 'reverse':
    kl = kl_from_distributions(
      p_logits=p_logits, 
      q_logits=q_logits, 
      p_loc=p_loc, 
      p_scale=p_scale, 
      q_loc=q_loc, 
      q_scale=q_scale, 
      logits_mask=logits_mask, 
    )
  else:
    raise NotImplementedError(f'Unknown kl {reg_type}')

  return kl

def kl_from_distributions(
  *, 
  p_logits=None, 
  q_logits=None, 
  p_loc=None, 
  q_loc=None, 
  p_scale=None,  
  q_scale=None,  
  logits_mask=None, 
):
  if p_logits is None:
    pd = distrax.MultivariateNormalDiag(p_loc, p_scale)
    qd = distrax.MultivariateNormalDiag(q_loc, q_scale)
  else:
    if logits_mask is not None:
      p_logits = jnp.where(logits_mask, p_logits, -float('inf'))
      q_logits = jnp.where(logits_mask, q_logits, -float('inf'))
    pd = distrax.Categorical(p_logits)
    qd = distrax.Categorical(q_logits)
  kl = pd.kl_divergence(qd)

  return kl

def kl_from_samples(
  *,
  logp, 
  logq, 
  sample_prob=None
):
  if sample_prob is None:
    sample_prob = 1.
  p = lax.exp(logp)
  q = lax.exp(logq)
  approx_kl = q * lax.stop_gradient(-p / q / sample_prob)
  return approx_kl

def reverse_kl_from_samples(
  *,
  logp,
  logq, 
  sample_prob=None
):
  if sample_prob is None:
    sample_prob = 1.
  log_ratio = logp - logq
  p = lax.exp(logp)
  approx_kl = p * lax.stop_gradient((log_ratio + 1) / sample_prob)
  return approx_kl

def js_from_samples(
  *,
  p,
  q, 
  sample_prob
):
  q = jnp.clip(q, 1e-10, 1)
  p_plus_q = jnp.clip(p+q, 1e-10, 1)
  approx_js = .5 * q * lax.stop_gradient(
    (lax.log(2.) + lax.log(q) -  lax.log(p_plus_q)) / sample_prob)
  return approx_js

def js_from_distributions(
  *,
  p_logits=None, 
  q_logits=None, 
  p_loc=None, 
  q_loc=None, 
  p_scale=None,  
  q_scale=None,  
  logits_mask=None
):
  if p_logits is None:
    # NOTE: mid computes a linear combination of two Gaussians
    # not the mixture Gaussians. For more details, refer to 
    # https://stats.stackexchange.com/a/8648
    # a significant consequence is that the linear combination of 
    # two identical Gaussians yields a new Gaussian with a different scale.
    # Therefore, the JS computed in this way is not 0 even though
    # p and q are identical.
    mid_loc = (p_loc + q_loc) / 2.
    mid_scale = lax.pow(lax.pow(p_scale, 2.)/4. + lax.pow(q_scale, 2.)/4., 1/2)
    js = .5 * (
      kl_from_distributions(
        p_loc=p_loc, q_loc=mid_loc, 
        p_scale=p_scale, q_scale=mid_scale)
      + kl_from_distributions(
        p_loc=q_loc, q_loc=mid_loc, 
        p_scale=q_scale, q_scale=mid_scale
      )
    )
  else:
    avg = (p_logits + q_logits) / 2
    js = .5 * (
      kl_from_distributions(p_logits=p_logits, q_logits=avg, logits_mask=logits_mask)
      + kl_from_distributions(p_logits=q_logits, q_logits=avg, logits_mask=logits_mask)
    )
  return js

def tv_from_samples(
  *, 
  p,
  q, 
  sample_prob
):
  approx_tv = .5 * lax.abs(p-q) / sample_prob
  return approx_tv

def tsallis_log(p, tsallis_q):
  p = jnp.clip(p, 1e-10, 1)
  if tsallis_q == 1:
    return lax.log(p)
  else:
    return (p**(1-tsallis_q) - 1) / (1 - tsallis_q)

def tsallis_exp(p, tsallis_q):
  if tsallis_q == 1:
    return lax.exp(p)
  else:
    return jnp.maximum(
      0, 1 + (1-tsallis_q) * p)**(1 / (1-tsallis_q))

def tsallis_from_samples(
  *, 
  p, 
  q, 
  sample_prob, 
  tsallis_q, 
):
  approx_tsallis = q * lax.stop_gradient(
    lax.sign(q-p) * p * q ** (-tsallis_q) / sample_prob)

  return approx_tsallis

def reverse_tsallis_from_samples(
  *, 
  p, 
  q, 
  sample_prob, 
  tsallis_q,
):
  logp = tsallis_log(p, tsallis_q)
  logq = tsallis_log(q, tsallis_q)
  approx_tsallis = p * lax.stop_gradient(
    -tsallis_q * p ** (tsallis_q - 1) * (logq - logp) / sample_prob)
  return approx_tsallis

def tsallis_from_distributions(
  *,
  pi1=None, 
  pi2=None,
  p_mean=None,  
  p_std=None,  
  q_mean=None,  
  q_std=None,  
  logits_mask=None,
  tsallis_q, 
):
  if pi1 is None:
    raise NotImplementedError('Tsallis divergence only support discrete probability distributions')
  else:
    log_pi1 = tsallis_log(jnp.clip(pi1, 1e-10, 1), tsallis_q)
    log_pi2 = tsallis_log(jnp.clip(pi2, 1e-10, 1), tsallis_q)
    log_ratio = log_pi1 - log_pi2
    if logits_mask is not None:
      log_ratio = jnp.where(logits_mask, log_ratio, 0)
    chex.assert_tree_all_finite(log_ratio)
    tsallis = jnp.sum(pi1**tsallis_q * log_ratio, axis=-1)

  return tsallis


def wasserstein(
  p_loc, p_scale, q_loc, q_scale
):
  return lax.pow(p_loc - q_loc, 2.) + lax.pow(p_scale - q_scale, 2.)
