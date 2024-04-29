import jax.numpy as jnp

from env.utils import get_action_mask
from core.elements.loss import LossBase
from core.typing import AttrDict
from .utils import *


class Loss(LossBase):
  def loss(
    self, 
    theta, 
    rng, 
    data,
    name='train', 
  ):
    stats = AttrDict()

    data.action_mask = get_action_mask(data.action)
    if data.sample_mask is not None:
      stats.n_alive_units = jnp.sum(data.sample_mask, -1)

    state = None if data.state is None else data.state.policy
    act_dists = compute_policy_dist(
      self.model, theta.policy, rng, data, state, bptt=self.config.prnn_bptt
    )

    if len(act_dists) == 1:
      act_dist = act_dists[DEFAULT_ACTION]
      pi_logprob = act_dist.log_prob(data.action[DEFAULT_ACTION])
    else:
      # assert set(action) == set(act_dists), (set(action), set(act_dists))
      for k, v in data.action.items():
        print(k, v.shape)
      pi_logprob = sum([ad.log_prob(a) for ad, a in 
                        zip(act_dists.values(), data.action.values())])
    loss = -jnp.mean(pi_logprob)
    stats.loss = loss

    return loss, stats


def create_loss(config, model, name='ppo'):
  loss = Loss(config=config, model=model, name=name)

  return loss
