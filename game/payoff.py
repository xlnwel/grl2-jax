import os
from typing import Dict, List, Tuple
import numpy as np

from core.ckpt import pickle
from core.log import do_logging
from core.typing import ModelPath, get_aid
from tools.utils import config_attr


class PayoffTableCheckpoint:
  def __init__(
    self, 
    step_size, 
    payoff_dir, 
    name='payoff', 
  ):
    self._step_size = step_size
    self._payoff_dir = payoff_dir
    self._name = name
    self._path = os.path.join(payoff_dir, f'{name}.pkl')

    self.payoffs = None
    self.counts = None

  """ Checkpoints """
  def save(self, to_print=True):
    data = {v: getattr(self, v) for v in vars(self) if not v.startswith('_')}
    pickle.save(
      data, filedir=self._payoff_dir, filename=self._name, 
      name='payoffs', to_print=to_print
    )

  def restore(self):
    data = pickle.restore(
      filedir=self._payoff_dir, filename=self._name, name='payoffs'
    )
    config_attr(self, data, filter_dict=False, config_as_attr=False)


class PayoffTable(PayoffTableCheckpoint):
  def __init__(
    self, 
    n_agents, 
    step_size, 
    payoff_dir, 
    name='payoff', 
  ):
    super().__init__(step_size, payoff_dir, name)
    self._n_agents = n_agents

    self.payoffs = [np.zeros([0] * n_agents, dtype=np.float32) * np.nan for _ in range(n_agents)]
    self.counts = [np.zeros([0] * n_agents, dtype=np.int64) for _ in range(n_agents)]

    self.restore()

  """ Payoff Retrieval """
  def get_payoffs(self, fill_nan=None):
    payoffs = []
    for p in self.payoffs:
      p = p.copy()
      if fill_nan is not None:
        assert isinstance(fill_nan, (int, float)), fill_nan
        p[np.isnan(p)] = fill_nan
      payoffs.append(p)
    return payoffs

  def get_payoffs_for_agent(self, aid: int, *, sid: int=None):
    """ Get the payoff table for agent aid """
    payoff = self.payoffs[aid]
    if sid is not None:
      payoff = payoff[(slice(None), ) * aid + (sid,)]
      assert len(payoff.shape) == self._n_agents - 1, (payoff.shape, self._n_agents)
    else:
      assert len(payoff.shape) == self._n_agents, (payoff.shape, self._n_agents)
    return payoff

  def get_counts(self):
    return self.counts

  def get_counts_for_agent(self, aid: int, *, sid: int):
    count = self.counts[aid]
    if sid is not None:
      count = count[(slice(None), ) * aid + (sid)]
      assert len(count.shape) == self._n_agents - 1, (count.shape, self._n_agents)
    else:
      assert len(count.shape) == self._n_agents, (count.shape, self._n_agents)
    return count

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    if from_scratch:
      self.payoffs = [
        np.zeros([0] * self._n_agents, dtype=np.float32) * np.nan
        for _ in range(self._n_agents)
      ]
      self.counts = [
        np.zeros([0] * self._n_agents, dtype=np.int64) 
        for _ in range(self._n_agents)
      ]
    else:
      self.payoffs = [np.zeros_like(p) * np.nan for p in self.payoffs]
      self.counts = [np.zeros_like(c) for c in self.counts]
    if name is not None:
      self._name = name

  def expand(self, aid):
    pad_width = [(0, 0) for _ in range(self._n_agents)]
    pad_width[aid] = (0, 1)
    for i in range(self._n_agents):
      self._expand(i, pad_width)

  def expand_all(self, aids: List[int]):
    assert len(aids) == self._n_agents, aids
    pad_width = [(0, 1) for _ in range(self._n_agents)]
    for aid in enumerate(aids):
      self._expand(aid, pad_width)

  def update(self, sids: Tuple[int], scores: List[List[float]]):
    """
    The i-th element in scores specifies the payoff result of sids 
    from the view of the i-th agent
    """
    assert len(sids) == self._n_agents, f'Some models are not specified: {sids}'
    for payoff, count, s in zip(self.payoffs, self.counts, scores):
      s_sum = sum(s)
      s_total = len(s)
      if s == []:
        continue
      elif count[sids] == 0:
        payoff[sids] = s_sum / s_total
      elif self._step_size == 0 or self._step_size is None:
        payoff[sids] = (count[sids] * payoff[sids] + s_sum) / (count[sids] + s_total)
      else:
        payoff[sids] += self._step_size * (s_sum / s_total - payoff[sids])
      assert not np.isnan(payoff[sids]), (count[sids], payoff[sids])
      count[sids] += s_total

  """ implementations """
  def _expand(self, aid, pad_width):
    self.payoffs[aid] = np.pad(self.payoffs[aid], pad_width, constant_values=np.nan)
    self.counts[aid] = np.pad(self.counts[aid], pad_width)


class PayoffTableWithModel(PayoffTable):
  def __init__(
    self, 
    n_agents, 
    step_size, 
    payoff_dir, 
    name='payoff', 
  ):
    # mappings between ModelPath and strategy index
    self.model2sid: List[Dict[ModelPath, int]] = [{} for _ in range(n_agents)]
    self.sid2model: List[List[ModelPath]] = [[] for _ in range(n_agents)]

    super().__init__(n_agents, step_size, payoff_dir, name=name)

  """ Get & Set """
  def get_all_models(self):
    return self.sid2model

  def get_sid2model(self):
    return self.sid2model
  
  def get_model2sid(self):
    return self.model2sid

  """ Payoff Retrieval """
  def get_payoffs_for_agent(self, aid: int, *, sid: int=None, model: ModelPath=None):
    if sid is None and model is not None:
      sid = self.model2sid[aid][model]
    payoff = super().get_payoffs_for_agent(aid, sid=sid)
    return payoff

  def get_counts_for_agent(self, aid: int, *, sid: int, model: ModelPath):
    if sid is None and model is not None:
      sid = self.model2sid[aid][model]
    counts = super().get_counts_for_agent(aid, sid=sid)
    return counts

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    super().reset(from_scratch, name=name)
    if from_scratch:
      self.model2sid: List[Dict[ModelPath, int]] = [{} for _ in range(self._n_agents)]
      self.sid2model: List[List[ModelPath]] = [[] for _ in range(self._n_agents)]

  def expand(self, model: ModelPath, aid=None):
    if aid is None:
      aid = get_aid(model.model_name)
    self._expand_mappings(aid, model)
    super().expand(aid)
    self._check_consistency(aid, model)

  def expand_all(self, models: List[ModelPath]):
    assert len(models) == self._n_agents, models
    pad_width = [(0, 1) for _ in range(self._n_agents)]
    for aid, model in enumerate(models):
      assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}'
      self._expand_mappings(aid, model)
      self._expand(aid, pad_width)

    for aid, model in enumerate(models):
      self._check_consistency(aid, model)

  def update(self, models: List[ModelPath], scores: List[List[float]]):
    assert len(models) == len(scores) == self._n_agents, (models, scores, self._n_agents)
    sids = tuple([
      m2sid[model] for m2sid, model in zip(self.model2sid, models) if model in m2sid
    ])
    super().update(sids, scores)
    # print('Payoffs', *self.payoffs, 'Counts', *self.counts, sep='\n')

  """ Implementations """
  def _expand_mappings(self, aid, model: ModelPath):
    if isinstance(model.model_name, str):
      assert aid == get_aid(model.model_name), (aid, model)
    else:
      assert isinstance(model.model_name, int), model.model_name
    assert model not in self.model2sid[aid], f'Model({model}) is already in {list(self.model2sid[aid])}'
    sid = self.payoffs[aid].shape[aid]
    self.model2sid[aid][model] = sid
    self.sid2model[aid].append(model)
    assert len(self.sid2model[aid]) == sid+1, (sid, self.sid2model)

  def _check_consistency(self, aid, model: ModelPath):
    if isinstance(model.model_name, str):
      assert aid == get_aid(model.model_name), (aid, model)
    else:
      assert isinstance(model.model_name, int), model.model_name
    for i in range(self._n_agents):
      assert self.payoffs[i].shape[aid] == self.model2sid[aid][model] + 1, \
        (self.payoffs[i].shape[aid], self.model2sid[aid][model] + 1)
      assert self.counts[i].shape[aid] == self.model2sid[aid][model] + 1, \
        (self.counts[i].shape[aid], self.model2sid[aid][model] + 1)


class SelfPlayPayoffTable(PayoffTableCheckpoint):
  def __init__(self, step_size, payoff_dir, name='payoff'):
    super().__init__(step_size, payoff_dir, name)
    self.payoffs = np.zeros([0] * 2, dtype=np.float32)
    self.counts = np.zeros([0] * 2, dtype=np.int64)

    self.restore()
  
  """ Payoff Retrieval """
  def get_payoffs(self, fill_nan=None, *, sid: int=None):
    payoffs = self.payoffs.copy()
    if fill_nan is not None:
      assert isinstance(fill_nan, (int, float)), fill_nan
      payoffs[np.isnan(payoffs)] = fill_nan
    if sid is not None:
      payoffs = payoffs[sid]
    return payoffs

  def get_counts(self, *, sid: int=None):
    counts = self.counts.copy()
    if sid is not None:
      counts = counts[sid]
    return counts

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    if from_scratch:
      self.payoffs = np.zeros([0] * 2, dtype=np.float32) * np.nan
      self.counts = np.zeros([0] * 2, dtype=np.int64) 
    else:
      self.payoffs = np.zeros_like(self.payoffs) * np.nan
      self.counts = np.zeros_like(self.counts)
    if name is not None:
      self._name = name

  def expand(self):
    pad_width = (0, 1)
    self._expand(pad_width)

  def update(self, sids: Tuple[int], scores: List[float]):
    assert len(sids) == 2, f'Strategies for both sides of agents were expected, but got {sids}'
    s_sum = sum(scores)
    s_total = len(scores)
    if sids[0] == sids[1]:
      self.payoffs[sids] = 0
    else:
      rsids = (sids[1], sids[0])
      if scores == []:
        return
      elif self.counts[sids] == 0:
        payoff = s_sum / s_total
        self.payoffs[sids] = payoff
        self.payoffs[rsids] = -payoff
      elif self._step_size == 0 or self._step_size is None:
        payoff = (self.counts[sids] * self.payoffs[sids] + s_sum) / (self.counts[sids] + s_total)
        self.payoffs[sids] = payoff
        self.payoffs[rsids] = -payoff
      else:
        payoff = s_sum / s_total
        new_payoff = self._step_size * (payoff - self.payoffs[sids])
        self.payoffs[sids] += new_payoff
        new_payoff = self._step_size * (-payoff - self.payoffs[rsids])
        self.payoffs[rsids] += new_payoff
      # assert self.payoffs[sids] + self.payoffs[rsids] == 1, (sids, self.payoffs[sids], rsids, self.payoffs[rsids])
      assert not np.isnan(self.payoffs[sids]), (self.counts[sids], self.payoffs[sids])
      self.counts[rsids] += s_total
    self.counts[sids] += s_total

  """ implementations """
  def _expand(self, pad_width):
    self.payoffs = np.pad(self.payoffs, pad_width, constant_values=np.nan)
    self.counts = np.pad(self.counts, pad_width)


class SelfPlayPayoffTableWithModel(SelfPlayPayoffTable):
  def __init__(self, step_size, payoff_dir, name='payoff'):
    # mappings between ModelPath and strategy index
    self.model2sid: Dict[ModelPath, int] = {}
    self.sid2model: List[ModelPath] = []

    super().__init__(step_size, payoff_dir, name)

  """ Get & Set """
  def get_all_models(self):
    return self.sid2model

  def get_sid2model(self):
    return self.sid2model
  
  def get_model2sid(self):
    return self.model2sid

  """ Payoff Retrieval """
  def get_payoffs(self, fill_nan=None, *, sid: int=None, model: ModelPath=None):
    if sid is None and model is not None:
      sid = self.model2sid[model]
    payoff = super().get_payoffs(fill_nan=fill_nan, sid=sid)
    return payoff

  def get_counts(self, *, sid: int=None, model: ModelPath=None):
    if sid is None and model is not None:
      sid = self.model2sid[model]
    counts = super().get_counts(sid=sid)
    return counts

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    super().reset(from_scratch, name=name)
    if from_scratch:
      self.model2sid: Dict[ModelPath, int] = {}
      self.sid2model: List[ModelPath] = []

  def expand(self, model: ModelPath):
    self._expand_mappings(model)
    super().expand()
    self._check_consistency(model)

  def update(self, models: List[ModelPath], scores: List[float]):
    assert len(models) == 2, models
    sids = tuple([self.model2sid[model] for model in models if model])
    super().update(sids, scores)
    # print('Payoffs', *self.payoffs, 'Counts', *self.counts, sep='\n')

  """ Implementations """
  def _expand_mappings(self, model: ModelPath):
    assert isinstance(model, ModelPath), model
    assert model not in self.model2sid, f'Model({model}) is already in {list(self.model2sid)}'
    sid = self.payoffs.shape[0]
    self.model2sid[model] = sid
    self.sid2model.append(model)
    assert len(self.sid2model) == sid+1, (sid, self.sid2model)

  def _check_consistency(self, model: ModelPath):
    assert self.payoffs.shape[0] == self.model2sid[model] + 1, \
      (self.payoffs.shape[0], self.model2sid[model] + 1)
    assert self.counts.shape[0] == self.model2sid[model] + 1, \
      (self.counts.shape[0], self.model2sid[model] + 1)
