import numpy as np

from tools.run import concat_along_unit_dim
from tools.rms import normalize, denormalize
from algo.ppo.elements.utils import compute_gae
from algo.ma_common.run import Runner


def prepare_buffer(
  agent, 
  env_output, 
  compute_return=True, 
):
  buffer = agent.buffer
  value = agent.compute_value(env_output)
  data = buffer.get_data({
    'value': value, 
    'state_reset': concat_along_unit_dim(env_output.reset)
  })
  data.raw_reward = data.reward
  data.reward = agent.actor.process_reward_with_rms(
    data.reward, data.discount, update_rms=True)
  obs = {k: data[k] for k in agent.actor.get_obs_names()}
  data.update({f'raw_{k}': v for k, v in obs.items()})
  data = agent.actor.normalize_obs(data, is_next=False)
  data = agent.actor.normalize_obs(data, is_next=True)
  agent.actor.update_obs_rms(obs)
  if 'sample_mask' not in data:
    data.sample_mask = np.ones_like(data.reward, np.float32)
  
  if compute_return:
    if agent.trainer.config.popart:
      poparts = [p.get_rms_stats(with_count=False, return_std=True) 
             for p in agent.trainer.popart]
      mean, std = [np.concatenate(s) for s in zip(*poparts)]
      value = denormalize(data.value, mean, std)
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
    if agent.trainer.config.popart:
      data.v_target = normalize(data.v_target, mean, std)

  buffer.move_to_queue(data)
