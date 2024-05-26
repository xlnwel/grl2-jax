from core.names import TRAIN_AXIS
from tools.utils import except_axis
from algo.ma_common.elements.utils import *


def norm_adv(
  config, 
  raw_adv, 
  teammate_log_ratio, 
  teammate_ratio_clip=None, 
  epsilon=1e-5
):
  if config.norm_adv:
    norm_adv = jax_math.standard_normalization(
      raw_adv, 
      zero_center=config.get('zero_center', True), 
      axis=except_axis(raw_adv, UNIT_DIM), 
      epsilon=epsilon, 
    )
  else:
    norm_adv = raw_adv
  jax_assert.assert_rank_compatibility(
    [norm_adv, teammate_log_ratio])
  tm_ratio = lax.exp(teammate_log_ratio)
  if teammate_ratio_clip is not None:
    tm_ratio = jnp.clip(tm_ratio, 1-teammate_ratio_clip, 1+teammate_ratio_clip)
  advantage = norm_adv * tm_ratio
  advantage = lax.stop_gradient(advantage)
  return norm_adv, advantage


def compute_target_adv(
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
  axis=TRAIN_AXIS.SEQ, 
):
  if config.adv_horizon:
    shape = reward.shape
    reward, discount, reset, value, next_value, ratio = \
      reshape_for_bptt(
        reward, discount, reset, value, next_value, ratio, bptt=config.adv_horizon
      )
    
  v_target, adv = jax_loss.compute_target_advantage(
    config=config, 
    reward=reward, 
    discount=discount, 
    reset=reset, 
    value=value, 
    next_value=next_value, 
    ratio=ratio, 
    gamma=gamma, 
    lam=lam, 
    axis=axis
  )

  if config.adv_horizon:
    v_target, adv = jax.tree_util.tree_map(
      lambda x: x.reshape(shape), (v_target, adv)
    )

  return v_target, adv
