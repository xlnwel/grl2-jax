import collections
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from tools.utils import batch_dicts
from replay import replay_registry


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
    value, next_value = value[:-1], value[1:]
  elif next_value.ndim < value.ndim:
    next_value = np.expand_dims(next_value, 1)
    next_value = np.concatenate([value[1:], next_value], 0)
  assert reward.shape == discount.shape == value.shape == next_value.shape, (reward.shape, discount.shape, value.shape, next_value.shape)
  
  delta = (reward + discount * gamma * next_value - value).astype(np.float32)
  discount = (discount if reset is None else (1 - reset)) * gae_discount
  
  next_adv = 0
  advs = np.zeros_like(reward, dtype=np.float32)
  for i in reversed(range(advs.shape[0])):
    advs[i] = next_adv = (delta[i] + discount[i] * next_adv)
  traj_ret = advs + value

  return advs, traj_ret


@replay_registry.register('tblocal')
class TurnBasedLocalBuffer(Buffer):
  def __init__(
    self, 
    config: AttrDict,
    env_stats: AttrDict,  
    model: Model,
    aid: int, 
    runner_id: int,
  ):
    super().__init__(config, env_stats, model, aid)

    self.runner_id = runner_id

    self.n_steps = self.config.n_steps
    self.n_envs = self.config.n_envs
    self.maxlen = self.n_envs * self.n_steps
    self.compute_return_at_once = self.config.get('compute_return_at_once', True)
    self.extract_next_info = self.config.get('extract_next_info', False)

    self.reset()

  def size(self):
    return self._size

  def is_empty(self):
    return self._size == 0

  def is_full(self):
    return self._size >= self.maxlen

  def reset(self):
    self._memory = []
    self._size = 0
    self._buffers = collections.defaultdict(lambda: collections.defaultdict(list))
    self._buff_lens = collections.defaultdict(int)

  def add(self, reset, **data):
    eids = data.pop('eid')
    uids = data.pop('uid')
    for i, (eid, uid) in enumerate(zip(eids, uids)):
      if reset[i]:
        assert self._buff_lens[(eid, uid)] == 0, (eid, uid, self._buff_lens[(eid, uid)])
        assert len(self._buffers[(eid, uid)]) == 0, (eid, uid, len(self._buffers[(eid, uid)])) 
      for k, v in data.items():
        self._buffers[(eid, uid)][k].append(v[i])

  def add_reward(self, eids, uids, reward, discount):
    assert len(eids) == len(uids), (eids, uids)
    for i, (eid, uid) in enumerate(zip(eids, uids)):
      if np.any(discount[i] == 0):
        # the last reward is added in func finish_episode for all units in the environment.
        np.testing.assert_equal(discount[i], 0)
        continue
      if not self._buffers[(eid, uid)]:
        # skip reward if no observation is received
        continue
      assert 'obs' in self._buffers[(eid, uid)], self._buffers[(eid, uid)]
      self._buffers[(eid, uid)]['reward'].append([reward[i][uid]])
      self._buff_lens[(eid, uid)] += 1

  def finish_episode(self, eids, uids, reward):
    for i, eid in enumerate(eids):
      for uid in uids:
        if self._buffers[(eid, uid)]:
          self._buffers[(eid, uid)]['reward'].append([reward[i][uid]])
          self._buff_lens[(eid, uid)] += 1
          self.merge_episode(eid, uid)
        else:
          self._reset_buffer(eid, uid)

  def merge_episode(self, eid, uid):
    episode = {k: np.stack(v) for k, v in self._buffers[(eid, uid)].items()}
    episode['discount'] = np.ones_like(episode['reward'], dtype=np.float32)
    episode['discount'][-1] = 0
    epslen = self._buff_lens[(eid, uid)]
    for k, v in episode.items():
      assert v.shape[0] == epslen, (k, v.shape, epslen)

    if self.compute_return_at_once:
      episode['advantage'], episode['v_target'] = compute_gae(
        reward=episode['reward'], 
        discount=episode['discount'],
        value=episode['value'],
        gamma=self.config.gamma,
        gae_discount=self.config.gamma * self.config.lam,
        next_value=np.array([0], np.float32), 
      )
    if self.extract_next_info:
      new_eps = {}
      for k, v in episode.items():
        if k in ['obs', 'global_state', 'action_mask', 'state_reset']:
          new_eps[f'next_{k}'] = v[1:]
        else:
          new_eps[k] = v[:-1]
      episode = new_eps
      epslen -= 1

    self._memory.append(episode)
    self._size += epslen
    self._reset_buffer(eid, uid)
    return episode

  def retrieve_all_data(self):
    data = batch_dicts(self._memory, np.concatenate)
    for k, v in data.items():
      assert v.shape[0] == self._size, (k, v.shape, self._size)
      v = v[-self.maxlen:]
      data[k] = np.reshape(v, (self.n_envs, self.n_steps, *v.shape[1:]))
    self.reset()
    return self.runner_id, data, self.maxlen

  def _reset_buffer(self, eid, uid):
    self._buffers[(eid, uid)] = collections.defaultdict(list)
    self._buff_lens[(eid, uid)] = 0
