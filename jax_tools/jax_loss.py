import jax
from jax import lax
import jax.numpy as jnp
import chex

from . import jax_assert, jax_div, jax_math, jax_utils


def to_loss(
  raw_stats, 
  coef, 
  mask=None, 
  n=None, 
  axis=None
):
  if coef is None:
    coef = 0
  scaled_loss = coef * raw_stats
  loss = jax_math.mask_mean(scaled_loss, mask, n, axis=axis)
  return scaled_loss, loss


def magic_box(logprob):
  return lax.exp(logprob - lax.stop_gradient(logprob))


def loaded_dice(logprob, lam, axis=1):
  dims, logprob = jax_utils.time_major(logprob, axis=axis)

  w = 0
  deps = []
  for lp in logprob:
    v = w * lam
    w = v + lp
    dep = magic_box(w) - magic_box(v)
    deps.append(dep)
  deps = jnp.asarray(deps)

  deps = jax_utils.undo_time_major(deps, dims, axis)

  return deps


def dice(logprob, lam, axis=1):
  """ The DiCE operator
  axis: the time dimension. If None, we do not consider the time dimension
  lam: the discount factor to reduce the effect of distant causal dependencies
    so as to trade-off bias and variance.
  """
  if axis is None:
    deps = magic_box(logprob)
  else:
    deps = loaded_dice(logprob, lam)
  
  return deps


def huber_loss(x, *, y=None, threshold=1.):
  if y is not None:   # if y is passed, take x-y as error, otherwise, take x as error
    x = x - y
  x = lax.abs(x)
  loss = jnp.where(x < threshold, 0.5 * lax.square(x), threshold * (x - 0.5 * threshold))

  return loss


def quantile_regression_loss(
  qtv, 
  target, 
  tau_hat, 
  kappa=1., 
  return_error=False
):
  assert qtv.shape[-1] == 1, qtv.shape
  assert target.shape[-2] == 1, target.shape
  assert tau_hat.shape[-1] == 1, tau_hat.shape
  chex.assert_rank([qtv, target, tau_hat])
  error = target - qtv       # [B, N, N']
  weight = lax.abs(tau_hat - jnp.asarray(error < 0, jnp.float32))   # [B, N, N']
  huber = huber_loss(error, threshold=kappa)          # [B, N, N']
  qr_loss = jnp.sum(jnp.mean(weight * huber, axis=-1), axis=-2) # [B]

  if return_error:
    return error, qr_loss
  return qr_loss


def h(x, epsilon=1e-3):
  """ h function defined in the transfomred Bellman operator 
  epsilon=1e-3 is used following recent papers(e.g., R2D2, MuZero, NGU)
  """
  sqrt_term = lax.sqrt(lax.abs(x) + 1.)
  return lax.sign(x) * (sqrt_term - 1.) + epsilon * x


def inverse_h(x, epsilon=1e-3):
  """ h^{-1} function defined in the transfomred Bellman operator
  epsilon=1e-3 is used following recent papers(e.g., R2D2, MuZero, NGU)
  """
  sqrt_term = lax.sqrt(1. + 4. * epsilon * (lax.abs(x) + 1. + epsilon))
  frac_term = (sqrt_term - 1.) / (2. * epsilon)
  return lax.sign(x) * (lax.square(frac_term) - 1.)


def retrace(
  *,
  reward, 
  q,  
  next_qs, 
  next_pi, 
  ratio, 
  discount, 
  reset=None, 
  gamma, 
  lam=.95, 
  c_clip=1, 
  axis=0, 
  tbo=False, 
  regularization=None, 
):
  """
  Params:
    discount = 1-done. 
    axis specifies the time dimension
  """
  chex.assert_equal_shape([reward, q, next_qs, discount, reset])

  # swap 'axis' with the 0-th dimension
  dims, (next_q, ratio, discount, reset) = \
    jax_utils.time_major(next_q, ratio, discount, reset, axis=axis)

  if tbo:
    next_qs = inverse_h(next_qs)
  next_v = jnp.sum(next_qs * next_pi, axis=-1)
  if regularization is not None:
    next_v -= regularization
  discount = discount * gamma
  delta = reward + discount * next_v - q
  next_c = lam * jnp.minimum(ratio[1:], c_clip)

  chex.assert_rank([delta[:-1], next_c])
  if reset is not None:
    discounted_ratio = (1 - reset[1:]) * gamma * next_c
  else:
    discounted_ratio = discount[1:] * next_c
  
  err = 0.
  errors = []
  for i in reversed(range(delta.shape[0])):
    err = delta[i] + discounted_ratio[i] * err
    errors.append(err)
  errors = jnp.array(errors[::-1])

  target = errors + q

  target = jax_utils.undo_time_major(target, dims=dims, axis=axis)

  if tbo:
    target = h(target)
    
  return target


def gae(
  *, 
  reward, 
  value, 
  next_value, 
  discount, 
  reset=None, 
  gamma=1, 
  lam=1, 
  axis=0
):
  jax_assert.assert_shape_compatibility([
    reward, value, next_value, discount, reset])
  
  # swap 'axis' with the 0-th dimension
  # to make all tensors time-major
  dims, (reward, value, next_value, discount, reset) = \
    jax_utils.time_major(reward, value, next_value, discount, reset, axis=axis)

  gae_discount = gamma * lam
  delta = reward + discount * gamma * next_value - value
  discount = (discount if reset is None else (1 - reset)) * gae_discount

  err = 0.
  advs = []
  for i in reversed(range(delta.shape[0])):
    err = delta[i] + discount[i] * err
    advs.append(err)
  advs = jnp.array(advs[::-1])
  vs = advs + value

  vs, advs = jax_utils.undo_time_major(vs, advs, dims=dims, axis=axis)

  return vs, advs


def v_trace(
  *,
  reward, 
  value, 
  next_value, 
  pi, 
  mu, 
  discount, 
  reset=None, 
  gamma=1, 
  lam=1, 
  c_clip=1, 
  rho_clip=1, 
  rho_clip_pg=1, 
  adv_type='vtrace', 
  axis=0
):
  """
  Params:
    discount = 1-done. 
    axis specifies the time dimension
  """
  ratio = pi / mu
  return v_trace_from_ratio(
    reward=reward, 
    value=value, 
    next_value=next_value,
    ratio=ratio, 
    discount=discount, 
    reset=reset, 
    gamma=gamma, 
    lam=lam, 
    c_clip=c_clip, 
    rho_clip=rho_clip, 
    rho_clip_pg=rho_clip_pg, 
    adv_type=adv_type, 
    axis=axis
  )


def v_trace_from_ratio(
  *, 
  reward, 
  value, 
  next_value, 
  ratio, 
  discount, 
  reset=None, 
  gamma=1, 
  lam=1, 
  c_clip=1, 
  rho_clip=1, 
  rho_clip_pg=1, 
  adv_type='vtrace', 
  axis=1
):
  """ This re-implementation of rlax.vtrace_td_error_and_advantage handles 
  infinite-horizon cases with hard reset.
  Params:
    discount = 1-done. 
    axis specifies the time dimension
  """
  if reward.ndim < ratio.ndim:
    chex.assert_rank(ratio, 4)
    ratio = jnp.prod(ratio, -1)
  jax_assert.assert_shape_compatibility(
    [reward, value, next_value, ratio, discount, reset])
  
  dims, (reward, value, next_value, ratio, discount, reset) = \
    jax_utils.time_major(
      reward, value, next_value, ratio, 
      discount, reset, axis=axis)

  clipped_c = jax_math.upper_clip(ratio, c_clip) * lam
  clipped_rho = jax_math.upper_clip(ratio, rho_clip)

  discount = discount * gamma
  delta = clipped_rho * (reward + discount * next_value - value)
  if reset is not None:
    discounted_ratio = (1 - reset) * gamma * clipped_c
  else:
    discounted_ratio = discount * clipped_c
  
  err = 0.
  errors = []
  for i in reversed(range(delta.shape[0])):
    err = delta[i] + discounted_ratio[i] * err
    errors.append(err)
  advs = jnp.array(errors[::-1])
  vs = advs + value

  if rho_clip_pg is None:
    clipped_rho_pg = 1.
  else:
    clipped_rho_pg = jax_math.upper_clip(ratio, rho_clip_pg)
  if adv_type == 'vtrace':
    # Following https://github.com/deepmind/rlax/blob/44ef3f04c8286bc9df51c85a0ec2475e85136294/rlax/_src/vtrace.py#L208
    # we use the lambda-mixture for the bootstrapped value
    next_vs = jnp.concatenate([
      lam * vs[1:] + (1-lam) * value[1:], 
      next_value[-1:]
    ], axis=0)
    vs = reward + discount * next_vs
    advs = clipped_rho_pg * (vs - value)
  elif adv_type == 'gae':
    advs = clipped_rho_pg * advs
  else:
    raise ValueError(adv_type)

  vs, advs = jax_utils.undo_time_major(vs, advs, dims=dims, axis=axis)
  
  return vs, advs


def compute_target_advantage(
  *, 
  config, 
  reward, 
  discount, 
  reset=None, 
  value, 
  next_value, 
  ratio, 
  gamma, 
  lam, 
  axis=1, 
):
  if config.target_type == 'vtrace':
    v_target, advantage = v_trace_from_ratio(
      reward=reward, 
      value=value, 
      next_value=next_value, 
      ratio=ratio, 
      discount=discount, 
      reset=reset, 
      gamma=gamma, 
      lam=lam, 
      c_clip=config.c_clip, 
      rho_clip=config.rho_clip, 
      rho_clip_pg=config.rho_clip_pg, 
      adv_type=config.get('adv_type', 'vtrace'), 
      axis=axis
    )
  elif config.target_type == 'gae':
    v_target, advantage = gae(
      reward=reward, 
      value=value, 
      next_value=next_value, 
      discount=discount, 
      reset=reset, 
      gamma=gamma, 
      lam=lam, 
      axis=axis
    )
  elif config.target_type == 'td':
    if reset is not None:
      discount = 1 - reset
    v_target = reward + discount * gamma * next_value
    advantage = v_target - value
  else:
    raise NotImplementedError

  return v_target, advantage


def pg_loss(
  *, 
  advantage, 
  logprob, 
):
  jax_assert.assert_shape_compatibility([advantage, logprob])
  pg = - advantage * logprob

  return pg


def is_pg_loss(
  *, 
  advantage, 
  ratio
):
  jax_assert.assert_shape_compatibility([advantage, ratio])
  loss = - advantage * ratio
  return loss


def entropy_loss(
  *, 
  entropy_coef, 
  entropy, 
  mask=None, 
  n=None
):
  jax_assert.assert_shape_compatibility([entropy, mask])
  scaled_entropy_loss, entropy_loss = to_loss(
    -entropy, 
    entropy_coef, 
    mask=mask, 
    n=n
  )

  return scaled_entropy_loss, entropy_loss


def ppo_loss(
  *, 
  advantage, 
  ratio, 
  clip_range, 
):
  jax_assert.assert_shape_compatibility([advantage, ratio])
  pg_loss, clipped_loss = _compute_ppo_policy_losses(
    advantage, ratio, clip_range)
  loss = jnp.maximum(pg_loss, clipped_loss)
  
  return pg_loss, clipped_loss, loss


def correct_ppo_loss(
  *, 
  advantage, 
  pi_logprob, 
  mu_logprob, 
  clip_range, 
  opt_pg=False
):
  ratio = jnp.exp(pi_logprob - mu_logprob)
  neutral_ratio = jnp.exp(pi_logprob - lax.stop_gradient(pi_logprob))
  jax_assert.assert_shape_compatibility([advantage, ratio])
  cond = advantage >= 0
  if opt_pg:
    cond = jnp.logical_and(cond, ratio < 1)
  ratio = jnp.where(cond, neutral_ratio, ratio)
  pg_loss, clipped_loss = _compute_ppo_policy_losses(
    advantage, ratio, clip_range)
  loss = jnp.maximum(pg_loss, clipped_loss)
  
  return pg_loss, clipped_loss, loss


def high_order_ppo_loss(
  *, 
  advantage, 
  ratio, 
  dice_op, 
  clip_range, 
):
  jax_assert.assert_shape_compatibility([advantage, ratio, dice_op])
  chex.assert_equal(dice_op, 1.)
  ratio = lax.stop_gradient(ratio)
  neg_adv = -advantage
  pg_loss = neg_adv * ratio * dice_op
  if clip_range is not None:
    dice_op = jnp.where(lax.abs(ratio - 1.) > clip_range, 1., dice_op)
  clipped_loss = neg_adv * dice_op * jnp.clip(
    ratio, 1. - clip_range, 1. + clip_range)
  loss = jnp.maximum(pg_loss, clipped_loss)

  return pg_loss, clipped_loss, loss


def joint_ppo_loss(
  *, 
  advantage, 
  ratio, 
  joint_ratio=None, 
  clip_range, 
  mask=None, 
  n=None,
):
  jax_assert.assert_shape_compatibility([ratio, mask])
  if mask is not None and n is None:
    mask = jnp.prod(mask, axis=-1)
  if joint_ratio is None:
    joint_ratio = jnp.prod(ratio, axis=-1)
  clipped_ratio = jnp.clip(ratio, 1. - clip_range, 1. + clip_range)
  joint_clipped_ratio = jnp.prod(clipped_ratio, axis=-1)
  jax_assert.assert_shape_compatibility([joint_ratio, advantage])
  neg_adv = -advantage
  pg_loss = neg_adv * joint_ratio
  clipped_loss = neg_adv * joint_clipped_ratio

  loss = jnp.maximum(pg_loss, clipped_loss)

  return pg_loss, clipped_loss, loss


def _compute_ppo_value_losses(
  value, 
  traj_ret, 
  old_value, 
  clip_range, 
  huber_threshold=None
):
  value_diff = value - old_value
  value_clipped = old_value + jnp.clip(value_diff, -clip_range, clip_range)
  if huber_threshold is None:
    loss1 = .5 * lax.square(value - traj_ret)
    loss2 = .5 * lax.square(value_clipped - traj_ret)
  else:
    loss1 = huber_loss(value, y=traj_ret, threshold=huber_threshold)
    loss2 = huber_loss(value_clipped, y=traj_ret, threshold=huber_threshold)

  return value_diff, loss1, loss2


def clipped_value_loss(
  value, 
  traj_ret, 
  old_value, 
  clip_range, 
  mask=None, 
  n=None, 
  huber_threshold=None,
):
  chex.assert_equal_shape([value, traj_ret, old_value])
  value_diff, loss1, loss2 = _compute_ppo_value_losses(
    value, traj_ret, old_value, clip_range, huber_threshold)
  
  value_loss = jnp.maximum(loss1, loss2)
  clip_frac = jax_math.mask_mean(jnp.abs(value_diff) > clip_range, mask, n)

  return value_loss, clip_frac


def _compute_ppo_policy_losses(advantages, ratio, clip_range):
  neg_adv = -advantages
  pg_loss = neg_adv * ratio
  if clip_range is None:
    clipped_loss = pg_loss
  else:
    clipped_loss = neg_adv * jnp.clip(ratio, 1. - clip_range, 1. + clip_range)
  return pg_loss, clipped_loss


def compute_kl_loss(
  *, 
  reg_type,
  kl_coef=None, 
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
  sample_mask=None,
  n=None, 
):
  """ Compute the KL divergence between p and q,
  where p is the distribution to be optimize and 
  q is the target distribution
  """
  if kl_coef is not None:
    kl = jax_div.kl_divergence(
      reg_type=reg_type, 
      logp=logp, 
      logq=logq, 
      sample_prob=sample_prob, 
      p_logits=p_logits, 
      q_logits=q_logits, 
      p_loc=p_loc, 
      p_scale=p_scale, 
      q_loc=q_loc, 
      q_scale=q_scale, 
      logits_mask=logits_mask, 
    )
    raw_kl_loss, kl_loss = to_loss(
      kl, 
      kl_coef, 
      mask=sample_mask, 
      n=n
    )
  else:
    kl = 0.
    raw_kl_loss = 0.
    kl_loss = 0.

  return kl, raw_kl_loss, kl_loss


def compute_js(
  *, 
  js_type, 
  js_coef, 
  p=None, 
  q=None,
  sample_prob=None, 
  pi1=None,
  pi2=None,
  logits_mask=None, 
  sample_mask=None, 
  n=None
):
  if js_coef is not None:
    if js_type == 'approx':
      js = jax_div.js_from_samples(
        p=p, 
        q=q, 
        sample_prob=sample_prob, 
      )
    elif js_type == 'exact':
      js = jax_div.js_from_distributions(
        pi1=pi1, pi2=pi2, logits_mask=logits_mask
      )
    else:
      raise NotImplementedError(f'Unknown JS type {js_type}')
    raw_js_loss, js_loss = to_loss(
      js, 
      js_coef, 
      mask=sample_mask, 
      n=n
    )
  else:
    js = 0,
    raw_js_loss = 0.
    js_loss = 0.
  
  return js, raw_js_loss, js_loss


def compute_tsallis(
  *, 
  tsallis_type,
  tsallis_coef, 
  tsallis_q, 
  p=None,
  q=None, 
  sample_prob=None, 
  pi1=None,
  pi2=None,
  p_loc=None,
  q_loc=None,
  p_scale=None,
  q_scale=None,
  logits_mask=None, 
  sample_mask=None,
  n=None, 
):
  if tsallis_coef is not None:
    if tsallis_type == 'forward_approx':
      tsallis = jax_div.tsallis_from_samples(
        p=p, 
        q=q,
        sample_prob=sample_prob, 
        tsallis_q=tsallis_q, 
      )
    elif tsallis_type == 'reverse_approx':
      tsallis = jax_div.reverse_tsallis_from_samples(
        p=q, 
        q=p,
        sample_prob=sample_prob, 
        tsallis_q=tsallis_q, 
      )
    elif tsallis_type == 'forward':
      tsallis = jax_div.tsallis_from_distributions(
        pi1=pi1, 
        pi2=pi2, 
        p_loc=p_loc, 
        p_scale=p_scale, 
        q_loc=q_loc, 
        q_scale=q_scale, 
        logits_mask=logits_mask, 
        tsallis_q=tsallis_q, 
      )
    elif tsallis_type == 'reverse':
      tsallis = jax_div.tsallis_from_distributions(
        pi1=pi2, 
        pi2=pi1, 
        p_loc=q_loc, 
        p_scale=q_scale, 
        q_loc=p_loc,
        q_scale=p_scale, 
        logits_mask=logits_mask, 
        tsallis_q=tsallis_q
      )
    else:
      raise NotImplementedError(f'Unknown Tsallis {tsallis_type}')
    raw_tsallis_loss, tsallis_loss = to_loss(
      tsallis, 
      tsallis_coef, 
      mask=sample_mask, 
      n=n
    )
  else:
    tsallis = 0.
    raw_tsallis_loss = 0.
    tsallis_loss = 0.

  return tsallis, raw_tsallis_loss, tsallis_loss


def mbpo_model_loss(
  mean, 
  logvar, 
  target, 
):
  """ Model loss from MBPO: 
  https://github.com/jannerm/mbpo/blob/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/mbpo/models/bnn.py#L581
  """
  inv_var = lax.exp(-logvar)

  mean_loss = jnp.mean((mean - target)**2 * inv_var, -1)
  var_loss = jnp.mean(logvar, -1)

  return mean_loss, var_loss
