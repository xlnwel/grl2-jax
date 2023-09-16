from typing import List
import numpy as np

from core.ckpt.pickle import save, restore
from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from core.typing import AttrDict
from tools.schedule import PiecewiseSchedule
from tools.utils import batch_dicts, yield_from_tree
from replay.local import NStepBuffer
from replay.ds.sum_tree import SumTree
from replay import replay_registry
from replay.mixin.rms import TemporaryRMS


@replay_registry.register('per')
class ProportionalPER(Buffer):
  """ Base class for PER, left in case one day I implement rank-based PER """
  def __init__(
    self, 
    config: AttrDict, 
    env_stats: AttrDict, 
    model: Model, 
    aid: int=0, 
  ):
    super().__init__(config, env_stats, model, aid)

    self.n_envs = self.config.n_runners * self.config.n_envs

    self.max_size = int(self.config.max_size)
    self.min_size = int(self.config.min_size)
    self.batch_size = self.config.batch_size
    self.n_recency = self.config.get('n_recency', self.min_size)
    self.n_steps = self.config.n_steps

    if self.config.directory:
      filedir = self.config.directory
    elif self.config.root_dir and self.config.model_name:
      filedir = '/'.join([self.config.root_dir, self.config.model_name])
    else:
      filedir = None
    self._filedir = filedir
    self._filename = self.config.filename if self.config.filename else 'uniform'

    self._memory = [None for _ in range(self.max_size)]
    self._is_full = False
    self._idx = 0

    if self.config.model_norm_obs:
      self.obs_rms = TemporaryRMS(self.config.get('obs_name', 'obs'), [0])
    self._tmp_bufs: List[NStepBuffer] = [
      NStepBuffer(config, env_stats, model, aid, 0) 
      for _ in range(self.n_envs)
    ]

    self._top_priority = 1.
    self._data_structure = SumTree(self.max_size)
    self._use_is_ratio = self.config.use_is_ratio
    self._alpha = self.config.get('alpha', 0)
    self._beta = self.config.get('beta', .4)
    if self.config.beta_scheduler:
      assert isinstance(self.config.beta_scheduler, list)
      self._beta_scheduler = PiecewiseSchedule(self.config.beta_scheduler)
    else:
      self._beta_scheduler = None
    self._sample_i = 0   # count how many times self._sample is called

  def __len__(self):
    if self._is_full:
      return self.max_size
    else:
      return self._idx

  def ready_to_sample(self):
    return len(self._memory) >= self.min_size

  def get_obs_rms(self):
    if self.config.model_norm_obs:
      return self.obs_rms.retrieve_rms()
  
  def reset(self):
    pass
  
  def collect_and_pop(self, **data):
    data = self._prepare_data(**data)
    return self.add_and_pop(**data)

  def add(self, **data):
    trajs = []
    if self.n_envs > 1:
      for i, d in enumerate(yield_from_tree(data)):
        if i >= len(self._tmp_bufs):
          self._tmp_bufs.append(NStepBuffer(
            self.config, self.env_stats, self.model, self.aid, 0))
        traj = self._tmp_bufs[i].add(**d)
        if traj is not None:
          trajs.append(traj)
    else:
      traj = self._tmp_bufs[0].add(**data)
      if traj is not None:
        trajs.append(traj)
    self.merge(trajs)

  def add_and_pop(self, **data):
    trajs = []
    popped_data = []
    if self.n_envs > 1:
      for i, d in enumerate(yield_from_tree(data)):
        if i >= len(self._tmp_bufs):
          self._tmp_bufs.append(NStepBuffer(
            self.config, self.env_stats, self.model, self.aid, 0))
        traj = self._tmp_bufs[i].add(**d)
        if traj is not None:
          trajs.append(traj)
    else:
      traj = self._tmp_bufs[0].add(**data)
      if traj is not None:
        trajs.append(traj)
    popped_data.extend(self.merge_and_pop(trajs))

    return popped_data

  def merge(self, trajs):
    if isinstance(trajs, dict):
      trajs = [trajs]
    n = len(trajs)
    idxes = self._get_next_idxes(self._idx, n)
    for i, traj in zip(idxes, trajs):
      self._memory[i] = traj
    assert len(self) <= self.max_size, len(self)
    self._update_obs_rms(trajs)

    priority = self._top_priority * np.ones(n)
    self.update_data_structure(idxes, priority)
    self._idx += n
    if self._idx >= self.max_size:
      self._idx %= self.max_size
      self._is_full = True

  def merge_and_pop(self, trajs):
    if isinstance(trajs, dict):
      trajs = [trajs]
    popped_data = []
    n = len(trajs)
    idxes = self._get_next_idxes(self._idx, n)
    for i, traj in zip(idxes, trajs):
      if len(self._memory) == self._memory.maxlen:
        popped_data.append(self._memory[i])
      self._memory[i] = traj
    self._update_obs_rms(trajs)
    
    priority = self._top_priority * np.ones(n)
    self.update_data_structure(idxes, priority)
    self._idx += n
    if self._idx >= self.max_size:
      self._idx %= self.max_size
      self._is_full = True

    return popped_data

  """ Sampling """
  def sample_from_recency(self, batch_size, sample_keys=None, n=None, add_seq_dim=False):
    batch_size = batch_size or self.batch_size
    n = max(batch_size, n or self.n_recency)
    if n > len(self):
      return None
    idxes = self._get_prev_idxes(self._idx, n)
    idxes = np.random.choice(idxes, size=batch_size, replace=False)

    samples = self._get_samples(
      idxes, self._memory, sample_keys=sample_keys, add_seq_dim=add_seq_dim)

    return samples

  def sample(self, batch_size=None):
    if self.ready_to_sample():
      samples = self._sample(batch_size=batch_size)
      self._sample_i += 1
      if self._beta_scheduler:
        self._update_beta()
    else:
      samples = None

    return samples

  def ergodic_sample(self, batch_size=None, n=None):
    if not self.ready_to_sample():
      return None
    batch_size = batch_size or self.batch_size
    n = len(self) if n is None else min(n, len(self))
    n = n // batch_size * batch_size
    idxes = self._get_prev_idxes(self._idx, n)
    np.random.shuffle(idxes)
    for i in range(0, n, batch_size):
      yield self._get_samples(idxes[i:i+batch_size], self._memory)

  def range_sample(self, start, n):
    if not self.ready_to_sample():
      return None
    end = min(len(self), start+n)
    idxes = np.arange(start, end)
    return self._get_samples(idxes, self._memory)

  def update_priorities(self, priorities, idxes):
    assert not np.any(np.isnan(priorities)), priorities
    np.testing.assert_array_less(0, priorities)
    self._top_priority = max(self._top_priority, np.max(priorities))
    self.update_data_structure(idxes, priorities)

  """ Retrieval """
  def retrieve_all_data(self):
    self.clear_local_buffer()
    data = self._memory
    self._memory = [None for _ in range(self.max_size)]
    self._is_full = False
    self._idx = 0
    return data

  def clear_local_buffer(self, drop_data=False):
    for b in self._tmp_bufs:
      if drop_data:
        b.reset()
      else:
        traj = b.retrieve_all_data()
        if traj:
          self.merge(traj)

  """ Implementation """
  def _update_beta(self):
    self._beta = self._beta_scheduler(self._sample_i)
  
  def update_data_structure(self, idxes, priorities):
    priorities = priorities ** self._alpha
    self._data_structure.batch_update(idxes, priorities)

  def _compute_IS_ratios(self, probabilities):
    """
    w = (N * p)**(-beta)
    max(w) = max(N * p)**(-beta) = (N * min(p))**(-beta)
    norm_w = w / max(w) = (N*p)**(-beta) / (N * min(p))**(-beta)
         = (min(p) / p)**beta
    """
    IS_ratios = (np.min(probabilities) / probabilities)**self._beta

    return IS_ratios

  def _get_prev_idxes(self, idx, n):
    return np.arange(idx-n, idx) % self.max_size

  def _get_next_idxes(self, idx, n):
    return np.arange(idx, idx+n) % self.max_size

  def _sample(self, batch_size=None):
    batch_size = batch_size or self.batch_size
    total_priorities = self._data_structure.total_priorities

    intervals = np.linspace(0, total_priorities, batch_size+1)
    values = np.random.uniform(intervals[:-1], intervals[1:])
    priorities, idxes = self._data_structure.batch_find(values)
    priorities = np.where(idxes >= len(self), self._top_priority, priorities)   # a lazy fix for the out-of-range find in sum tree
    idxes = np.where(idxes >= len(self), self._idx-1, idxes)   # a lazy fix for the out-of-range find in sum tree
    # assert np.max(idxes) < len(self), f'idxes: {idxes}\nvalues: {values}\npriorities: {priorities}\ntotal: {total_priorities}, len: {len(self)}'
    # assert np.min(priorities) > 0, f'idxes: {idxes}\nvalues: {values}\npriorities: {priorities}\ntotal: {total_priorities}, len: {len(self)}'

    # compute importance sampling ratios
    samples = self._get_samples(idxes, self._memory)
    samples.idxes = idxes
    samples.priority = priorities
    if self._use_is_ratio:
      probabilities = priorities / total_priorities
      is_ratio = self._compute_IS_ratios(probabilities)
      samples.is_ratio = is_ratio.astype(np.float32)

    return samples

  def _get_samples(self, idxes, memory, sample_keys=None, add_seq_dim=True):
    if add_seq_dim:
      fn = lambda x: np.expand_dims(np.stack(x), 1)
    else:
      fn = np.stack
    samples = batch_dicts(
      [memory[i] for i in idxes], func=fn, keys=sample_keys
    )

    return samples

  def _update_obs_rms(self, trajs):
    if self.config.model_norm_obs:
      self.obs_rms.update_obs_rms(trajs)

  """ Save & Restore """
  def save(self, filedir=None, filename=None):
    filedir = filedir or self._filedir
    filename = filename or self._filename
    save(
      (self._memory, self._data_structure, self._is_full, self._idx, self._top_priority), 
      filedir=filedir, 
      filename=filename, 
      name='data'
    )
    do_logging(f'Number of transitions saved: {len(self)}')
  
  def restore(self, filedir=None, filename=None):
    filedir = filedir or self._filedir
    filename = filename or self._filename
    self._memory, self._is_full, self._idx, self._top_priority = \
      restore(
        filedir=filedir, 
        filename=filename, 
        default=([None for _ in range(self.max_size)], SumTree(self.max_size), False, 0, 1.), 
        name='data'
      )
    do_logging(f'Number of transitions restored: {len(self)}')
