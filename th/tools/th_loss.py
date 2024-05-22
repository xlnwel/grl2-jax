import torch

from th.tools import th_math, th_utils


def to_loss(raw_stats, coef, mask=None, axis=None):
  if coef is None:
    coef = 0
  scaled_loss = coef * raw_stats
  loss = th_math.mask_mean(scaled_loss, mask, axis=axis)
  return scaled_loss, loss


def huber_loss(x, *, y=None, threshold=1.):
  if y is not None:   # if y is passed, take x-y as error, otherwise, take x as error
    x = x - y
  x = torch.abs(x)
  loss = torch.where(
    x < threshold, 
    0.5 * torch.square(x), 
    threshold * (x - 0.5 * threshold))

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
  error = target - qtv       # [B, N, N']
  weight = torch.abs(tau_hat - (error < 0))   # [B, N, N']
  huber = huber_loss(error, threshold=kappa)          # [B, N, N']
  qr_loss = (weight * huber).mean(-1).sum(-2) # [B]

  if return_error:
    return error, qr_loss
  return qr_loss


def h(x, epsilon=1e-3):
  """ h function defined in the transfomred Bellman operator 
  epsilon=1e-3 is used following recent papers(e.g., R2D2, MuZero, NGU)
  """
  sqrt_term = torch.sqrt(torch.abs(x) + 1.)
  return torch.sign(x) * (sqrt_term - 1.) + epsilon * x


def inverse_h(x, epsilon=1e-3):
  """ h^{-1} function defined in the transfomred Bellman operator
  epsilon=1e-3 is used following recent papers(e.g., R2D2, MuZero, NGU)
  """
  sqrt_term = torch.sqrt(1. + 4. * epsilon * (torch.abs(x) + 1. + epsilon))
  frac_term = (sqrt_term - 1.) / (2. * epsilon)
  return torch.sign(x) * (torch.square(frac_term) - 1.)


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
  # swap 'axis' with the 0-th dimension
  dims, (next_q, ratio, discount, reset) = \
    th_utils.time_major(next_q, ratio, discount, reset, axis=axis)

  if tbo:
    next_qs = inverse_h(next_qs)
  next_v = (next_qs * next_pi).sum(-1)
  if regularization is not None:
    next_v -= regularization
  discount = discount * gamma
  delta = reward + discount * next_v - q
  next_c = lam * torch.minimum(ratio[1:], c_clip)

  if reset is not None:
    discounted_ratio = (1 - reset[1:]) * gamma * next_c
  else:
    discounted_ratio = discount[1:] * next_c
  
  err = 0.
  errors = []
  for i in reversed(range(delta.shape[0])):
    err = delta[i] + discounted_ratio[i] * err
    errors.append(err)
  errors = torch.tensor(errors[::-1])

  target = errors + q

  target = th_utils.undo_time_major(target, dims=dims, axis=axis)

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
  # swap 'axis' with the 0-th dimension
  # to make all tensors time-major
  dims, (reward, value, next_value, discount, reset) = \
    th_utils.time_major(reward, value, next_value, discount, reset, axis=axis)

  gae_discount = gamma * lam
  delta = reward + discount * gamma * next_value - value
  discount = (discount if reset is None else (1 - reset)) * gae_discount

  err = 0.
  advs = []
  for i in reversed(range(delta.shape[0])):
    err = delta[i] + discount[i] * err
    advs.append(err)
  advs = torch.stack(advs[::-1])
  vs = advs + value

  vs, advs = th_utils.undo_time_major(vs, advs, dims=dims, axis=axis)

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
    ratio = torch.prod(ratio, -1)

  dims, (reward, value, next_value, ratio, discount, reset) = \
    th_utils.time_major(
      reward, value, next_value, ratio, 
      discount, reset, axis=axis)

  clipped_c = th_math.upper_clip(ratio, c_clip) * lam
  clipped_rho = th_math.upper_clip(ratio, rho_clip)

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
  advs = torch.tensor(errors[::-1])
  vs = advs + value

  if rho_clip_pg is None:
    clipped_rho_pg = 1.
  else:
    clipped_rho_pg = th_math.upper_clip(ratio, rho_clip_pg)
  if adv_type == 'vtrace':
    # Following https://github.com/deepmind/rlax/blob/44ef3f04c8286bc9df51c85a0ec2475e85136294/rlax/_src/vtrace.py#L208
    # we use the lambda-mixture for the bootstrapped value
    next_vs = torch.cat([
      lam * vs[1:] + (1-lam) * value[1:], 
      next_value[-1:]
    ], axis=0)
    vs = reward + discount * next_vs
    advs = clipped_rho_pg * (vs - value)
  elif adv_type == 'gae':
    advs = clipped_rho_pg * advs
  else:
    raise ValueError(adv_type)

  vs, advs = th_utils.undo_time_major(vs, advs, dims=dims, axis=axis)
  
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


def pg_loss(*, advantage, logprob):
  pg = - advantage * logprob

  return pg


def is_pg_loss(*, advantage, ratio):
  loss = - advantage * ratio
  return loss


def entropy_loss(*, entropy_coef, entropy, mask=None):
  scaled_entropy_loss, entropy_loss = to_loss(
    -entropy, entropy_coef, mask=mask, 
  )

  return scaled_entropy_loss, entropy_loss


def ppo_loss(*, advantage, ratio, clip_range):
  pg_loss, clipped_loss = _compute_ppo_policy_losses(
    advantage, ratio, clip_range)
  loss = torch.maximum(pg_loss, clipped_loss)
  
  return pg_loss, clipped_loss, loss


def correct_ppo_loss(
  *, 
  advantage, 
  pi_logprob, 
  mu_logprob, 
  clip_range, 
  opt_pg=False
):
  ratio = torch.exp(pi_logprob - mu_logprob)
  neutral_ratio = torch.exp(pi_logprob - torch.stop_gradient(pi_logprob))
  cond = advantage >= 0
  if opt_pg:
    cond = torch.logical_and(cond, ratio < 1)
  ratio = torch.where(cond, neutral_ratio, ratio)
  pg_loss, clipped_loss = _compute_ppo_policy_losses(
    advantage, ratio, clip_range)
  loss = torch.maximum(pg_loss, clipped_loss)
  
  return pg_loss, clipped_loss, loss


def joint_ppo_loss(
  *, 
  advantage, 
  ratio, 
  joint_ratio=None, 
  clip_range, 
  mask=None, 
):
  if mask is not None and n is None:
    mask = torch.prod(mask, axis=-1)
  if joint_ratio is None:
    joint_ratio = torch.prod(ratio, axis=-1)
  clipped_ratio = ratio.clamp(1. - clip_range, 1. + clip_range)
  joint_clipped_ratio = torch.prod(clipped_ratio, axis=-1)
  neg_adv = -advantage
  pg_loss = neg_adv * joint_ratio
  clipped_loss = neg_adv * joint_clipped_ratio

  loss = torch.maximum(pg_loss, clipped_loss)

  return pg_loss, clipped_loss, loss


def _compute_ppo_value_losses(
  value, 
  traj_ret, 
  old_value, 
  clip_range, 
  huber_threshold=None
):
  value_diff = value - old_value
  value_clipped = old_value + value_diff.clamp(-clip_range, clip_range)
  if huber_threshold is None:
    loss1 = .5 * torch.square(value - traj_ret)
    loss2 = .5 * torch.square(value_clipped - traj_ret)
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
  huber_threshold=None,
):
  value_diff, loss1, loss2 = _compute_ppo_value_losses(
    value, traj_ret, old_value, clip_range, huber_threshold)
  
  value_loss = torch.maximum(loss1, loss2)
  clip_frac = th_math.mask_mean(
    (torch.abs(value_diff) > clip_range).to(torch.float32), mask)

  return value_loss, clip_frac


def _compute_ppo_policy_losses(advantages, ratio, clip_range):
  neg_adv = -advantages
  pg_loss = neg_adv * ratio
  if clip_range is None:
    clipped_loss = pg_loss
  else:
    clipped_loss = neg_adv * ratio.clamp(1. - clip_range, 1. + clip_range)
  return pg_loss, clipped_loss


def mbpo_model_loss(
  mean, 
  logvar, 
  target, 
):
  """ Model loss from MBPO: 
  https://github.com/jannerm/mbpo/blob/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/mbpo/models/bnn.py#L581
  """
  inv_var = torch.exp(-logvar)

  mean_loss = ((mean - target)**2 * inv_var).mean(-1)
  var_loss = logvar.mean(-1)

  return mean_loss, var_loss
