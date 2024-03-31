from jax import lax, random
import jax.numpy as jnp

from env.utils import get_action_mask
from core.names import TRAIN_AXIS
from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_loss
from tools.rms import denormalize, normalize
from .utils import *


class Loss(LossBase):
  def loss(self, theta, rng, data,name='train'):
    data = data.copy()
    rngs = random.split(rng, 2)
    stats = dict2AttrDict(self.config.stats, to_copy=True)

    if data.sample_mask is not None:
      stats.n_alive_units = jnp.sum(data.sample_mask, -1)

    stats.value, next_value = compute_values(
      self.model.forward_value, theta.value, rngs[0], data, 
      None if data.state is None else data.state.value, 
      bptt=self.config.vrnn_bptt, seq_axis=TRAIN_AXIS.SEQ, 
    )

    data.action_mask = get_action_mask(data.action)
    data.state_reset = data.state_reset[:, :-1] if 'state_reset' in data else None
    state = None if data.state is None else data.state.policy
    act_dists, stats.pi_logprob, stats.log_ratio, stats.ratio = compute_policy(
      self.model, theta.policy, rngs[1], data, state, self.config.prnn_bptt)
    stats = record_policy_stats(data, stats, act_dists)

    if 'advantage' in data:
      stats.raw_adv = data.pop('advantage')
      stats.raw_v_target = data.pop('v_target')
    else:
      if self.config.popart:
        value = lax.stop_gradient(denormalize(
          stats.value, data.popart_mean, data.popart_std))
        next_value = denormalize(
          next_value, data.popart_mean, data.popart_std)
      else:
        value = lax.stop_gradient(stats.value)

      stats.raw_v_target, stats.raw_adv = jax_loss.compute_target_advantage(
        config=self.config, 
        reward=data.reward, 
        discount=data.discount, 
        reset=data.reset, 
        value=value, 
        next_value=next_value, 
        ratio=lax.stop_gradient(stats.ratio), 
        gamma=stats.gamma, 
        lam=stats.lam, 
        axis=TRAIN_AXIS.SEQ
      )
    if self.config.popart:
      stats.v_target = normalize(stats.raw_v_target, data.popart_mean, data.popart_std)
    else:
      stats.v_target = stats.raw_v_target
    stats.v_target = lax.stop_gradient(stats.v_target)
    stats = record_target_adv(stats)
    stats.advantage = norm_adv(
      self.config, 
      stats.raw_adv, 
      sample_mask=data.sample_mask, 
      n=data.n, 
      epsilon=self.config.get('epsilon', 1e-5)
    )

    actor_loss, stats = compute_actor_loss(
      self.config, data, stats, act_dists, stats.entropy_coef
    )

    value_loss, stats = compute_vf_loss(self.config, data, stats)
    stats = summarize_adv_ratio(stats, data)
    loss = actor_loss + value_loss
    stats.loss = loss

    return loss, stats

  def value_loss(self, theta, rng, policy_theta, data, name='train/value'):
    data = data.copy()
    rngs = random.split(rng, 2)
    stats = dict2AttrDict(self.config.stats, to_copy=True)

    if data.sample_mask is not None:
      stats.n_alive_units = jnp.sum(data.sample_mask, -1)

    stats.value, next_value = compute_values(
      self.model.forward_value, theta, rngs[0], data, 
      None if data.state is None else data.state.value, 
      bptt=self.config.vrnn_bptt, seq_axis=TRAIN_AXIS.SEQ, 
    )

    data.action_mask = get_action_mask(data.action)
    data.state_reset = data.state_reset[:, :-1] if 'state_reset' in data else None
    state = None if data.state is None else data.state.policy
    _, _, _, ratio = compute_policy(
      self.model, policy_theta, rngs[1], data, state, self.config.prnn_bptt
    )

    if 'advantage' in data:
      stats.raw_adv = data.pop('advantage')
      stats.raw_v_target = data.pop('v_target')
    else:
      if self.config.popart:
        value = lax.stop_gradient(denormalize(
          stats.value, data.popart_mean, data.popart_std))
        next_value = denormalize(
          next_value, data.popart_mean, data.popart_std)
      else:
        value = lax.stop_gradient(stats.value)

      stats.raw_v_target, stats.raw_adv = jax_loss.compute_target_advantage(
        config=self.config, 
        reward=data.reward, 
        discount=data.discount, 
        reset=data.reset, 
        value=value, 
        next_value=next_value, 
        ratio=lax.stop_gradient(ratio), 
        gamma=stats.gamma, 
        lam=stats.lam, 
        axis=TRAIN_AXIS.SEQ
      )
    if self.config.popart:
      stats.v_target = normalize(stats.raw_v_target, data.popart_mean, data.popart_std)
    else:
      stats.v_target = stats.raw_v_target
    stats.v_target = lax.stop_gradient(stats.v_target)
    stats = record_target_adv(stats)

    value_loss, stats = compute_vf_loss(self.config, data, stats)
    stats.advantage = norm_adv(
      self.config, 
      stats.raw_adv, 
      sample_mask=data.sample_mask, 
      n=data.n, 
      epsilon=self.config.get('epsilon', 1e-5)
    )
    loss = value_loss

    return loss, stats

  def policy_loss(self, theta, rng, data, stats, name='train/policy'):
    data = data.copy()
    data.action_mask = get_action_mask(data.action)
    data.state_reset = data.state_reset[:, :-1] if 'state_reset' in data else None
    state = None if data.state is None else data.state.policy
    act_dists, stats.pi_logprob, stats.log_ratio, stats.ratio = \
      compute_policy(self.model, theta, rng, data, state, self.config.prnn_bptt)
    stats = record_policy_stats(data, stats, act_dists)

    actor_loss, stats = compute_actor_loss(
      self.config, data, stats, act_dists, stats.entropy_coef
    )

    stats = summarize_adv_ratio(stats, data)
    loss = actor_loss

    return loss, stats


def create_loss(config, model, name='ppo'):
  loss = Loss(config=config, model=model, name=name)

  return loss
