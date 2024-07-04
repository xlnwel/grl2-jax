from jax import lax, random

from jx.elements.loss import Loss as LossBase
from core.typing import dict2AttrDict
from .utils import *


class Loss(LossBase):
  def q_loss(
    self, 
    theta, 
    rng, 
    policy_params, 
    target_qs_params, 
    temp_params, 
    data, 
    name='train/q', 
  ):
    rngs = random.split(rng, 4)
    stats = dict2AttrDict(self.config.stats, to_copy=True)

    next_data = dict2AttrDict({
      'obs': data.next_obs, 
      'state_reset': data.next_state_reset, 
      'state': data.next_state, 
      'action_mask': data.next_action_mask, 
    })
    next_action, next_logprob, _ = compute_action_logprob(
      self.model, policy_params, rngs[0], next_data, bptt=self.config.prnn_bptt
    )

    next_qs_state = None if data.next_state is None else data.next_state.qs
    next_q = compute_qs(
      self.modules.Q, 
      target_qs_params, 
      rngs[1], 
      data.next_global_state, 
      next_action, 
      data.next_state_reset, 
      next_qs_state, 
      bptt=self.config.qrnn_bptt, 
      return_minimum=True
    )
    _, temp = self.modules.temp(temp_params, rngs[2])
    q_target = compute_target(
      data.reward, 
      data.discount, 
      stats.gamma, 
      next_q, 
      temp, 
      next_logprob
    )
    q_target = lax.stop_gradient(q_target)
    stats.q_target = q_target

    qs_state = None if data.state is None else data.state.qs
    qs = compute_qs(
      self.modules.Q, 
      theta, 
      rngs[3], 
      data.global_state, 
      data.action, 
      data.state_reset, 
      qs_state, 
      bptt=self.config.qrnn_bptt
    )
    loss, stats = compute_q_loss(
      self.config, qs, q_target, data, stats
    )

    return loss, stats

  def policy_loss(
    self, 
    theta, 
    rng, 
    qs_params, 
    temp_params, 
    data, 
    stats, 
    name='train/policy', 
  ):
    if not stats:
      stats = dict2AttrDict(self.config.stats, to_copy=True)
    rngs = random.split(rng, 3)

    actions, logprobs, act_dists = compute_action_logprob(
      self.model, theta, rngs[0], data, bptt=self.config.prnn_bptt
    )
    for k in logprobs:
      stats[f'{k}_logprob'] = logprobs[k]
      stats.update(act_dists[k].get_stats(f'{k}_pi'))
      stats[f'{k}_entropy'] = act_dists[k].entropy()
    logprob = sum(logprobs.values())
    stats.logprob = logprob

    qs_state = None if data.state is None else data.state.qs
    q = compute_qs(
      self.modules.Q, 
      qs_params, 
      rngs[1], 
      data.global_state, 
      actions, 
      data.state_reset, 
      qs_state, 
      bptt=self.config.qrnn_bptt, 
      return_minimum=True
    )
    stats.q = q
    _, temp = self.modules.temp(temp_params, rngs[2])
    loss, stats = compute_policy_loss(
      self.config, q, logprob, temp, data, stats
    )

    return loss, stats

  def temp_loss(
    self, 
    theta, 
    rng, 
    stats, 
  ):
    log_temp, temp = self.modules.temp(theta, rng)
    act_dim = sum(self.model.config.policy.action_dim.values())
    target_entropy = self.config.get('target_entropy', -act_dim)
    loss, stats = compute_temp_loss(temp, log_temp, target_entropy, stats)

    return loss, stats


def create_loss(config, model, name='sac'):
  loss = Loss(config=config, model=model, name=name)

  return loss
