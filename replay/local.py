from abc import abstractmethod
import logging
import collections
import math
import numpy as np

from jx.elements.buffer import Buffer
from tools.utils import batch_dicts, stack_data_with_state
from replay.utils import *

logger = logging.getLogger(__name__)


class LocalBuffer(Buffer):
  def __init__(
    self, 
    config, 
    env_stats, 
    model, 
    aid, 
    runner_id,
  ):
    super().__init__(config, env_stats, model, aid)
    self.runner_id = runner_id

    self._buffer = []

    self._add_attributes()
  
  def _add_attributes(self):
    pass

  def name(self):
    return self.config.type

  def __len__(self):
    return len(self._buffer)

  def is_full(self):
    return len(self._buffer) >= self.config.n_steps

  @abstractmethod
  def retrieve_all_data(self):
    raise NotImplementedError

  @abstractmethod
  def add(self, **data):
    raise NotImplementedError


class EpisodicBuffer(LocalBuffer):
  def reset(self):
    self._buffer.clear()
    self._size = 0

  def retrieve_all_data(self):
    data = [stack_data_with_state(d, seq_axis=0) for d in self._buffer]
    self.reset()

    return data

  def add(self, **data):
    self._buffer.append(data)

    if np.all(data['reset']) or np.all(data['discount'] == 0):
      eps = batch_dicts(self._buffer)
      self._buffer = []
    else:
      eps = None
    
    return eps


class NStepBuffer(LocalBuffer):
  def _add_attributes(self):
    self.max_steps = self.config.get('max_steps', 1)
    self.gamma = self.config.gamma
    self._buffer = collections.deque(maxlen=self.max_steps)

  def is_full(self):
    return len(self._buffer) == self.max_steps

  def reset(self):
    self._buffer.clear()

  def retrieve_all_data(self):
    data = list(self._buffer)
    self.reset()

    return data

  def add(self, **data):
    """ Add experience to local memory """
    if self.max_steps == 1:
      return [data]
    if self.is_full():
      result = [self._buffer.popleft()]
    else:
      result = None
    reward = data['reward']
    for i, d in enumerate(reversed(self._buffer)):
      d['reward'] += self.gamma**(i+1) * reward
      d['steps'] += 1
      for k, v in data.items():
        if k.startswith('next_'):
          d[k] = v
    data['steps'] = np.ones_like(reward)
    self._buffer.append(data)
    if data['discount'] == 0:
      result = self.retrieve_all_data()
    return result


class VecEnvNStepBuffer(NStepBuffer):
  """ Local memory only stores one episode of transitions from n environments """
  def reset(self):
    assert self.is_full(), self._idx
    self._idx = self._extra_len
    for v in self._buffer.values():
      v[:, :self._extra_len] = v[:, self._seqlen:]

  def add(self, env_ids=None, **data):
    """ Add experience to local memory """
    if self._buffer == {}:
      # initialize memory
      init_buffer(self._buffer, pre_dims=(self._n_envs, self._memlen), 
            has_steps=self._extra_len>1, **data)
      print_buffer(self._buffer, 'Local Buffer')

    idx = self._idx
    
    for k, v in data.items():
      if isinstance(self._buffer[k], np.ndarray):
        self._buffer[k][:, idx] = v
      else:
        for i in range(self._n_envs):
          self._buffer[k][i][idx] = v[i]
    if self._extra_len > 1:
      self._buffer['steps'][:, idx] = 1

    self._idx += 1

  def retrieve_all_data(self):
    assert self.is_full(), self._idx
    return self.retrieve(self._seqlen)
  
  def retrieve(self, seqlen=None):
    seqlen = seqlen or self._idx
    results = adjust_n_steps_envvec(self._buffer, seqlen, 
      self._n_steps, self._max_steps, self._gamma)
    value = None
    for k, v in results.items():
      if k in ('q', 'v'):
        value = results[k]
        pass
      else:
        results[k] = v[:, :seqlen].reshape(-1, *v.shape[2:])
    if value:
      idx = np.broadcast_to(np.arange(seqlen), (self._n_envs, seqlen))
      results['q'] = value[idx]
      results['next_q'] = value[idx + results.get('steps', 1)]
    if 'mask' in results:
      mask = results.pop('mask')
      results = {k: v[mask] for k, v in results.items()}
    if 'steps' in results:
      results['steps'] = results['steps'].astype(np.float32)

    return results


class SequentialBuffer(LocalBuffer):
  def reset(self):
    self._idx = self._memlen - self._reset_shift

  def _add_attributes(self):
    if not hasattr(self, '_reset_shift'):
      self._reset_shift = getattr(self, '_burn_in_size', 0) or self._sample_size
    self._extra_len = 1
    self._memlen = self._sample_size + self._extra_len

  def add(self, **data):
    if self._buffer == {}:
      for k in data:
        if k in self._state_keys:
          self._buffer[k] = collections.deque(
            maxlen=math.ceil(self._memlen / self._reset_shift))
        else:
          self._buffer[k] = collections.deque(maxlen=self._memlen)

    for k, v in data.items():
      if k not in self._state_keys or self._idx % self._reset_shift == 0:
        self._buffer[k].append(v)
    
    self._idx += 1
  
  def clear(self):
    self._idx = 0


class EnvSequentialBuffer(SequentialBuffer):
  def retrieve_all_data(self):
    assert self.is_full(), self._idx
    results = {}
    for k, v in self._buffer.items():
      if k in self._state_keys:
        results[k] = v[0]
      elif k in self._extra_keys:
        results[k] = np.array(v)
      else:
        results[k] = np.array(v)[:self._sample_size]

    return results


class VecEnvSequentialBuffer(SequentialBuffer):
  def retrieve_all_data(self):
    assert self.is_full(), self._idx
    results = {}
    for k, v in self._buffer.items():
      if k in self._state_keys:
        results[k] = v[0]
      elif k in self._extra_keys:
        results[k] = np.swapaxes(np.array(v), 0, 1)
      else:
        results[k] = np.swapaxes(np.array(list(v)[:self._sample_size]), 0, 1)
    
    results = [{k: v[i] for k, v in results.items()} for i in range(self._n_envs)] 
    for seq in results:
      for k, v in seq.items():
        if k in self._state_keys:
          pass
        elif k in self._extra_keys:
          assert v.shape[0] == self._sample_size + self._extra_len, (k, v.shape)
        else:
          assert v.shape[0] == self._sample_size, (k, v.shape)
    assert len(results) == self._n_envs, results
    
    return results
