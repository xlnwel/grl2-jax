import numpy as np

from th.tools.th_utils import to_numpy
from algo.ma_common.run import Runner


def compute_gae(reward, discount, value, gamma, gae_discount, 
                next_value=None, reset=None):
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


def prepare_buffer(
  agent, 
  env_output, 
  compute_return=True, 
):
  buffer = agent.buffer
  value = agent.compute_value(env_output)
  data = buffer.get_data({
    'value': value, 
    'state_reset': env_output.reset
  })
  data.raw_reward = data.reward
  data.reward = agent.actor.process_reward_with_rms(
    data.reward, data.discount, update_rms=True)
  obs = {k: data[k] for k in agent.actor.get_obs_names()}
  data.update({f'raw_{k}': v for k, v in obs.items()})
  data = agent.actor.normalize_obs(data, is_next=False)
  data = agent.actor.normalize_obs(data, is_next=True)
  if 'sample_mask' not in data:
    data.sample_mask = np.ones_like(data.reward, np.float32)
  agent.actor.update_obs_rms(
    obs, mask=data.sample_mask, feature_mask=agent.env_stats.feature_mask)
  
  if compute_return:
    if agent.trainer.config.popart:
      value = agent.actor.model.vnorm.denormalize(data.value)
      value = to_numpy(value)
    else:
      value = data.value
    value, next_value = value[:, :-1], value[:, 1:]
    data.advantage, data.v_target = compute_gae(
      reward=data.reward, 
      discount=data.discount,
      value=value,
      gamma=buffer.config.gamma,
      gae_discount=buffer.config.gamma * buffer.config.lam,
      next_value=next_value, 
      reset=data.reset,
    )

  buffer.move_to_queue(data)
