import numpy as np
import jax
from jax import lax
import jax.numpy as jnp

from core.names import DEFAULT_ACTION, PATH_SPLIT
from core.typing import AttrDict
from tools.utils import expand_dims_match, expand_shape_match
from jax_tools import jax_assert, jax_math, jax_loss, jax_utils


UNIT_DIM = 2

def get_initial_state(state, i):
  return jax.tree_util.tree_map(lambda x: x[:, i], state)


def reshape_for_bptt(*args, bptt):
  return jax.tree_util.tree_map(
    lambda x: None if x is None else x.reshape(-1, bptt, *x.shape[2:]), args
  )


def compute_values(
  func, 
  params, 
  rng, 
  x, 
  next_x, 
  state_reset, 
  state, 
  bptt, 
  seq_axis=1,
  **kwargs
):
  if state is None:
    value, _ = func(params, rng, x)
    if next_x is None:
      next_value = None
    else:
      next_value, _ = func(params, rng, next_x) 
  else:
    state_reset, next_state_reset = jax_utils.split_data(
      state_reset, axis=seq_axis)
    if bptt is not None:
      shape = x.shape[:-1]
      assert x.shape[1] % bptt == 0, (x.shape, bptt)
      x, next_x, state_reset, next_state_reset, state = \
        reshape_for_bptt(
          x, next_x, state_reset, next_state_reset, state, bptt=bptt
        )
    state0 = get_initial_state(state, 0)
    value, _ = func(params, rng, x, state_reset, state0)
    if next_x is None:
      next_value = None
    else:
      state1 = get_initial_state(state, 1)
      next_value, _ = func(params, rng, next_x, next_state_reset, state1)
    if bptt is not None:
      value, next_value = jax.tree_util.tree_map(
        lambda x: x if x is None else x.reshape(shape), (value, next_value)
      )
  if next_value is not None:
    next_value = lax.stop_gradient(next_value)
    jax_assert.assert_shape_compatibility([value, next_value])

  return value, next_value


def compute_policy_dist(
  model, 
  params, 
  rng, 
  x, 
  state_reset, 
  state, 
  action_mask=None, 
  bptt=None
):
  if state is not None and bptt is not None:
    shape = x.shape[:-1]
    assert x.shape[1] % bptt == 0, (x.shape, bptt)
    x, state_reset, state, action_mask = reshape_for_bptt(
      x, state_reset, state, action_mask, bptt=bptt
    )
  state = get_initial_state(state, 0)
  act_outs, _ = model.modules.policy(
    params, rng, x, state_reset, state, action_mask=action_mask
  )
  if state is not None and bptt is not None:
    act_outs = jax.tree_util.tree_map(
      lambda x: x.reshape(*shape, -1) if x.ndim > len(shape) else x, act_outs
    )
  act_dists = model.policy_dist(act_outs)
  return act_dists


def compute_policy(
  model, 
  params, 
  rng, 
  x, 
  action, 
  mu_logprob, 
  state_reset, 
  state, 
  action_mask=None, 
  bptt=None, 
):
  act_dists = compute_policy_dist(
    model, params, rng, x, state_reset, state, action_mask, bptt=bptt)
  if len(act_dists) == 1:
    act_dist = act_dists[DEFAULT_ACTION]
    pi_logprob = act_dist.log_prob(action[DEFAULT_ACTION])
  else:
    # assert set(action) == set(act_dists), (set(action), set(act_dists))
    pi_logprob = sum([ad.log_prob(action[k]) for k, ad in act_dists.items()])
  jax_assert.assert_shape_compatibility([pi_logprob, mu_logprob])
  log_ratio = pi_logprob - mu_logprob
  ratio = lax.exp(log_ratio)
  return act_dists, pi_logprob, log_ratio, ratio


def prefix_name(terms, name):
  if name is not None:
    new_terms = AttrDict()
    for k, v in terms.items():
      if PATH_SPLIT not in k:
        new_terms[f'{name}{PATH_SPLIT}{k}'] = v
      else:
        new_terms[k] = v
    return new_terms
  return terms


def compute_gae(
  reward, 
  discount, 
  value, 
  gamma, 
  gae_discount, 
  next_value=None, 
  reset=None, 
):
  if next_value is None:
    value, next_value = value[:, :-1], value[:, 1:]
  elif next_value.ndim < value.ndim:
    next_value = np.expand_dims(next_value, 1)
    next_value = np.concatenate([value[:, 1:], next_value], 1)
  assert reward.shape == discount.shape == value.shape == next_value.shape, (reward.shape, discount.shape, value.shape, next_value.shape)
  
  delta = (reward + discount * gamma * next_value - value).astype(np.float32)
  discount = (discount if reset is None else (1 - reset)) * gae_discount
  
  next_adv = 0
  advs = np.zeros_like(reward, dtype=np.float32)
  for i in reversed(range(advs.shape[1])):
    advs[:, i] = next_adv = (delta[:, i] + discount[:, i] * next_adv)
  traj_ret = advs + value

  return advs, traj_ret


def compute_actor_loss(
  config, 
  data, 
  stats, 
  act_dists, 
  entropy_coef, 
):
  if config.get('policy_sample_mask', True):
    sample_mask = data.sample_mask
  else:
    sample_mask = None
  if stats.advantage.ndim < stats.ratio.ndim:
    stats.advantage = expand_shape_match(stats.advantage, stats.ratio, np=jnp)

  if config.pg_type == 'pg':
    raw_pg_loss = jax_loss.pg_loss(
      advantage=stats.advantage, 
      logprob=stats.pi_logprob, 
    )
  elif config.pg_type == 'is':
    raw_pg_loss = jax_loss.is_pg_loss(
      advantage=stats.advantage, 
      ratio=stats.ratio, 
    )
  elif config.pg_type == 'ppo':
    ppo_pg_loss, ppo_clip_loss, raw_pg_loss = \
      jax_loss.ppo_loss(
        advantage=stats.advantage, 
        ratio=stats.ratio, 
        clip_range=config.ppo_clip_range, 
      )
    stats.ppo_pg_loss = ppo_pg_loss
    stats.ppo_clip_loss = ppo_clip_loss
  elif config.pg_type == 'correct_ppo':
    cppo_pg_loss, cppo_clip_loss, raw_pg_loss = \
      jax_loss.correct_ppo_loss(
        advantage=stats.advantage, 
        pi_logprob=stats.pi_logprob, 
        mu_logprob=data.mu_logprob, 
        clip_range=config.ppo_clip_range, 
        opt_pg=config.opt_pg
      )
    stats.cppo_pg_loss = cppo_pg_loss
    stats.cppo_clip_loss = cppo_clip_loss
  else:
    raise NotImplementedError
  if raw_pg_loss.ndim == 4:   # reduce the action dimension for continuous actions
    raw_pg_loss = jnp.sum(raw_pg_loss, axis=-1)
  scaled_pg_loss, pg_loss = jax_loss.to_loss(
    raw_pg_loss, 
    stats.pg_coef, 
    mask=sample_mask, 
    n=data.n
  )
  stats.raw_pg_loss = raw_pg_loss
  stats.scaled_pg_loss = scaled_pg_loss
  stats.pg_loss = pg_loss

  if len(act_dists) == 1:
    entropy = act_dists[DEFAULT_ACTION].entropy()
  entropy = {k: ad.entropy() for k, ad in act_dists.items()}
  for k, v in entropy.items():
    stats[f'{k}_entropy'] = v
  entropy = sum(entropy.values())
  scaled_entropy_loss, entropy_loss = jax_loss.entropy_loss(
    entropy_coef=entropy_coef, 
    entropy=entropy, 
    mask=sample_mask, 
    n=data.n
  )
  stats.entropy = entropy
  stats.scaled_entropy_loss = scaled_entropy_loss
  stats.entropy_loss = entropy_loss

  loss = pg_loss + entropy_loss
  stats.actor_loss = loss

  if sample_mask is not None:
    sample_mask = expand_shape_match(sample_mask, stats.ratio, np=jnp)
  clip_frac = jax_math.mask_mean(
    lax.abs(stats.ratio - 1.) > config.get('ppo_clip_range', .2), 
    sample_mask, data.n)
  stats.clip_frac = clip_frac
  stats.approx_kl = jax_math.mask_mean(
    .5 * (data.mu_logprob - stats.pi_logprob)**2, sample_mask, data.n
  )

  return loss, stats


def compute_vf_loss(
  config, 
  data, 
  stats, 
):
  if config.get('value_sample_mask', False):
    sample_mask = data.sample_mask
  else:
    sample_mask = None
  
  v_target = stats.v_target

  value_loss_type = config.value_loss
  if value_loss_type == 'huber':
    raw_value_loss = jax_loss.huber_loss(
      stats.value, 
      y=v_target, 
      threshold=config.huber_threshold
    )
  elif value_loss_type == 'mse':
    raw_value_loss = .5 * (stats.value - v_target)**2
  elif value_loss_type == 'clip' or value_loss_type == 'clip_huber':
    old_value, _ = jax_utils.split_data(
      data.value, data.next_value, axis=1
    )
    raw_value_loss, stats.v_clip_frac = jax_loss.clipped_value_loss(
      stats.value, 
      v_target, 
      old_value, 
      config.value_clip_range, 
      huber_threshold=config.huber_threshold, 
      mask=sample_mask, 
      n=data.n,
    )
  else:
    raise ValueError(f'Unknown value loss type: {value_loss_type}')
  stats.raw_value_loss = raw_value_loss
  scaled_value_loss, value_loss = jax_loss.to_loss(
    raw_value_loss, 
    coef=stats.value_coef, 
    mask=sample_mask, 
    n=data.n
  )
  
  stats.scaled_value_loss = scaled_value_loss
  stats.value_loss = value_loss

  return value_loss, stats


def record_target_adv(stats):
  stats.explained_variance = jax_math.explained_variance(
    stats.v_target, stats.value)
  # stats.v_target_unit_std = jnp.std(stats.v_target, axis=-1)
  # stats.raw_adv_unit_std = jnp.std(stats.raw_adv, axis=-1)
  return stats


def record_policy_stats(data, stats, act_dists):
  # stats.diff_frac = jax_math.mask_mean(
  #   lax.abs(stats.pi_logprob - data.mu_logprob) > 1e-5, 
  #   data.sample_mask, data.n)
  # stats.approx_kl = .5 * jax_math.mask_mean(
  #   (stats.log_ratio)**2, data.sample_mask, data.n)
  # stats.approx_kl_max = jnp.max(.5 * (stats.log_ratio)**2)
  if len(act_dists) == 1:
    stats.update(act_dists[DEFAULT_ACTION].get_stats(prefix='pi'))
  else:
    for k, ad in act_dists.items():
      k.replace('action_', '')
      stats.update(ad.get_stats(prefix=f'{k}_pi'))

  return stats


def summarize_adv_ratio(stats, data):
  # if stats.raw_adv.ndim < stats.ratio.ndim:
  #   raw_adv = expand_dims_match(stats.raw_adv, stats.ratio)
  # else:
  #   raw_adv = stats.raw_adv
  # stats.raw_adv_ratio_pp = jnp.logical_and(raw_adv > 0, stats.ratio > 1)
  # stats.raw_adv_ratio_pn = jnp.logical_and(raw_adv > 0, stats.ratio < 1)
  # stats.raw_adv_ratio_np = jnp.logical_and(raw_adv < 0, stats.ratio > 1)
  # stats.raw_adv_ratio_nn = jnp.logical_and(raw_adv < 0, stats.ratio < 1)
  # stats.raw_adv_zero = raw_adv == 0
  # stats.ratio_one = stats.ratio == 1
  # stats.adv_ratio_pp = jnp.logical_and(stats.advantage > 0, stats.ratio > 1)
  # stats.adv_ratio_pn = jnp.logical_and(stats.advantage > 0, stats.ratio < 1)
  # stats.adv_ratio_np = jnp.logical_and(stats.advantage < 0, stats.ratio > 1)
  # stats.adv_ratio_nn = jnp.logical_and(stats.advantage < 0, stats.ratio < 1)
  # stats.adv_zero = stats.advantage == 0
  # stats.pn_ratio = jnp.where(stats.adv_ratio_pn, stats.ratio, 0)
  # stats.np_ratio = jnp.where(stats.adv_ratio_np, stats.ratio, 0)

  return stats
