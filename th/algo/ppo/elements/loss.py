from envs.utils import get_action_mask
from th.core.names import TRAIN_AXIS
from th.core.elements.loss import LossBase
from th.core.typing import dict2AttrDict
from th.tools import th_loss
from .utils import *


class Loss(LossBase):
  def loss(self, data):
    data = data.copy()
    stats = dict2AttrDict(self.config.stats, to_copy=True)

    if data.sample_mask is not None:
      stats.n_alive_units = data.sample_mask.sum(-1)

    stats.value, next_value = compute_values(
      self.model.forward_value, data, 
      None if data.state is None else data.state.value, 
      bptt=self.config.vrnn_bptt, seq_axis=TRAIN_AXIS.SEQ, 
    )

    data.action_mask = get_action_mask(data.action)
    data.state_reset = data.state_reset[:, :-1] if 'state_reset' in data else None
    state = None if data.state is None else data.state.policy
    act_dists, stats.pi_logprob, stats.log_ratio, stats.ratio = compute_policy(
      self.model, data, state, self.config.prnn_bptt)
    stats = record_policy_stats(data, stats, act_dists)

    if 'advantage' in data:
      stats.raw_adv = data.pop('advantage')
      stats.raw_v_target = data.pop('v_target')
    else:
      if self.config.popart:
        value = self.model.vnorm.denormalize(stats.value).detach()
        next_value = self.model.vnorm.denormalize(next_value).detach()
      else:
        value = stats.value.detach()

      stats.raw_v_target, stats.raw_adv = th_loss.compute_target_advantage(
        config=self.config, 
        reward=data.reward, 
        discount=data.discount, 
        reset=data.reset, 
        value=value, 
        next_value=next_value, 
        ratio=stats.ratio.detach(), 
        gamma=stats.gamma, 
        lam=stats.lam, 
        axis=TRAIN_AXIS.SEQ
      )
    if self.config.popart:
      stats.v_target = self.model.vnorm.normalize(stats.raw_v_target)
    else:
      stats.v_target = stats.raw_v_target
    stats.v_target = stats.v_target.detach()
    stats = record_target_adv(stats)
    stats.advantage = norm_adv(
      self.config, 
      stats.raw_adv, 
      sample_mask=data.sample_mask, 
      epsilon=self.config.get('epsilon', 1e-5)
    )

    actor_loss, stats = compute_actor_loss(
      self.config, data, stats, act_dists, stats.entropy_coef
    )

    value_loss, stats = compute_vf_loss(self.config, data, stats)
    stats = summarize_adv_ratio(stats, data)
    loss = actor_loss + value_loss
    stats.loss = loss

    return actor_loss, value_loss, stats


def create_loss(config, model, name='ppo'):
  loss = Loss(config=config, model=model, name=name)

  return loss
