import os
from typing import Dict, List, Tuple
import numpy as np

from tools import pickle
from tools.log import do_logging
from core.ckpt.base import PickleCheckpointBase
from core.typing import ModelPath, get_aid
from tools.utils import config_attr


class PayoffTableCheckpoint(PickleCheckpointBase):
  def __init__(
    self, 
    step_size, 
    dir, 
    name='payoff', 
  ):
    self.step_size = step_size
    self.dir = dir
    self.name = name
    self.path = os.path.join(dir, f'{name}.pkl')

    self._payoffs = None
    self._counts = None

  """ Checkpoints """
  def retrieve(self):
    data = {v[1:]: getattr(self, v) for v in vars(self) if v.startswith('_')}
    return data

  def load(self, data):
    config_attr(self, data, filter_dict=False, private_attr=True)

  def save(self, to_print=True):
    data = {v[1:]: getattr(self, v) for v in vars(self) if v.startswith('_')}
    pickle.save(
      data, filedir=self.dir, filename=self.name, 
      name='payoffs', to_print=to_print
    )

  def restore(self, to_print=True):
    data = pickle.restore(
      filedir=self.dir, filename=self.name, 
      name='payoffs', to_print=to_print
    )
    self.load(data)


class PayoffTable(PayoffTableCheckpoint):
  def __init__(
    self, 
    n_agents, 
    step_size, 
    dir, 
    name='payoff', 
  ):
    super().__init__(step_size, dir, name)
    self.n_agents = n_agents

    self._payoffs = [np.zeros([0] * n_agents, dtype=np.float32) * np.nan for _ in range(n_agents)]
    self._counts = [np.zeros([0] * n_agents, dtype=np.int64) for _ in range(n_agents)]
  
  def size(self, aid: int):
    return self._payoffs[aid].shape[0]

  """ Payoff Retrieval """
  def get_payoffs(self, fill_nan=None):
    payoffs = []
    for p in self._payoffs:
      p = p.copy()
      if fill_nan is not None:
        assert isinstance(fill_nan, (int, float)), fill_nan
        p[np.isnan(p)] = fill_nan
      payoffs.append(p)
    return payoffs

  def get_payoffs_for_agent(self, aid: int, *, sid: int=None):
    """ Get the payoff table for agent aid """
    payoff = self._payoffs[aid]
    if sid is not None:
      payoff = payoff[(slice(None), ) * aid + (sid,)]
      assert len(payoff.shape) == self.n_agents - 1, (payoff.shape, self.n_agents)
    else:
      assert len(payoff.shape) == self.n_agents, (payoff.shape, self.n_agents)
    return payoff

  def get_counts(self):
    return self._counts

  def get_counts_for_agent(self, aid: int, *, sid: int):
    count = self._counts[aid]
    if sid is not None:
      count = count[(slice(None), ) * aid + (sid)]
      assert len(count.shape) == self.n_agents - 1, (count.shape, self.n_agents)
    else:
      assert len(count.shape) == self.n_agents, (count.shape, self.n_agents)
    return count

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    if from_scratch:
      self._payoffs = [
        np.zeros([0] * self.n_agents, dtype=np.float32) * np.nan
        for _ in range(self.n_agents)
      ]
      self._counts = [
        np.zeros([0] * self.n_agents, dtype=np.int64) 
        for _ in range(self.n_agents)
      ]
    else:
      self._payoffs = [np.zeros_like(p) * np.nan for p in self._payoffs]
      self._counts = [np.zeros_like(c) for c in self._counts]
    if name is not None:
      self.name = name

  def expand(self, aid):
    pad_width = [(0, 0) for _ in range(self.n_agents)]
    pad_width[aid] = (0, 1)
    for i in range(self.n_agents):
      self._expand(i, pad_width)

  def expand_all(self, aids: List[int]):
    assert len(aids) == self.n_agents, aids
    pad_width = [(0, 1) for _ in range(self.n_agents)]
    for aid in enumerate(aids):
      self._expand(aid, pad_width)

  def update(self, sids: Tuple[int], scores: List[List[float]]):
    """
    The i-th element in scores specifies the payoff result of sids 
    from the view of the i-th agent
    """
    assert len(sids) == self.n_agents, f'Some models are not specified: {sids}'
    for payoff, count, s in zip(self._payoffs, self._counts, scores):
      s_sum = sum(s)
      s_total = len(s)
      if s == []:
        continue
      elif count[sids] == 0:
        payoff[sids] = s_sum / s_total
      elif self.step_size == 0 or self.step_size is None:
        payoff[sids] = (count[sids] * payoff[sids] + s_sum) / (count[sids] + s_total)
      else:
        payoff[sids] += self.step_size * (s_sum / s_total - payoff[sids])
      assert not np.isnan(payoff[sids]), (count[sids], payoff[sids])
      count[sids] += s_total

  """ implementations """
  def _expand(self, aid, pad_width):
    self._payoffs[aid] = np.pad(self._payoffs[aid], pad_width, constant_values=np.nan)
    self._counts[aid] = np.pad(self._counts[aid], pad_width)


class PayoffTableWithModel(PayoffTable):
  def __init__(
    self, 
    n_agents, 
    step_size, 
    dir, 
    name='payoff', 
  ):
    # mappings between ModelPath and strategy index
    self._model2sid: List[Dict[ModelPath, int]] = [{} for _ in range(n_agents)]
    self._sid2model: List[List[ModelPath]] = [[] for _ in range(n_agents)]

    super().__init__(n_agents, step_size, dir, name=name)

  def __contains__(self, aid_model: Tuple[int, ModelPath]):
    aid, model = aid_model
    return model in self._sid2model[aid]

  """ Get & Set """
  def get_all_models(self):
    return self._sid2model

  def get_sid2model(self):
    return self._sid2model
  
  def get_model2sid(self):
    return self._model2sid

  """ Payoff Retrieval """
  def get_payoffs_for_agent(self, aid: int, *, sid: int=None, model: ModelPath=None):
    if sid is None and model is not None:
      sid = self._model2sid[aid][model]
    payoff = super().get_payoffs_for_agent(aid, sid=sid)
    return payoff

  def get_counts_for_agent(self, aid: int, *, sid: int, model: ModelPath):
    if sid is None and model is not None:
      sid = self._model2sid[aid][model]
    counts = super().get_counts_for_agent(aid, sid=sid)
    return counts

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    super().reset(from_scratch, name=name)
    if from_scratch:
      self._model2sid: List[Dict[ModelPath, int]] = [{} for _ in range(self.n_agents)]
      self._sid2model: List[List[ModelPath]] = [[] for _ in range(self.n_agents)]

  def expand(self, model: ModelPath, aid=None):
    if aid is None:
      aid = get_aid(model.model_name)
    self._expand_mappings(aid, model)
    super().expand(aid)
    self._check_consistency(aid, model)

  def expand_all(self, models: List[ModelPath]):
    assert len(models) == self.n_agents, models
    pad_width = [(0, 1) for _ in range(self.n_agents)]
    for aid, model in enumerate(models):
      assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}'
      self._expand_mappings(aid, model)
      self._expand(aid, pad_width)

    for aid, model in enumerate(models):
      self._check_consistency(aid, model)

  def update(self, models: List[ModelPath], scores: List[List[float]]):
    assert len(models) == len(scores) == self.n_agents, (models, scores, self.n_agents)
    sids = tuple([
      m2sid[model] for m2sid, model in zip(self._model2sid, models) if model in m2sid
    ])
    super().update(sids, scores)
    # print('Payoffs', *self._payoffs, 'Counts', *self._counts, sep='\n')

  """ Implementations """
  def _expand_mappings(self, aid, model: ModelPath):
    if isinstance(model.model_name, str):
      assert aid == get_aid(model.model_name), (aid, model)
    else:
      assert isinstance(model.model_name, int), model.model_name
    assert model not in self._model2sid[aid], f'Model({model}) is already in {list(self._model2sid[aid])}'
    sid = self._payoffs[aid].shape[aid]
    self._model2sid[aid][model] = sid
    self._sid2model[aid].append(model)
    assert len(self._sid2model[aid]) == sid+1, (sid, self._sid2model)

  def _check_consistency(self, aid, model: ModelPath):
    if isinstance(model.model_name, str):
      assert aid == get_aid(model.model_name), (aid, model)
    else:
      assert isinstance(model.model_name, int), model.model_name
    for i in range(self.n_agents):
      assert self._payoffs[i].shape[aid] == self._model2sid[aid][model] + 1, \
        (self._payoffs[i].shape[aid], self._model2sid[aid][model] + 1)
      assert self._counts[i].shape[aid] == self._model2sid[aid][model] + 1, \
        (self._counts[i].shape[aid], self._model2sid[aid][model] + 1)


class SelfPlayPayoffTable(PayoffTableCheckpoint):
  def __init__(self, step_size, dir, name='payoff'):
    super().__init__(step_size, dir, name)
    self._payoffs = np.zeros([0] * 2, dtype=np.float32)
    self._counts = np.zeros([0] * 2, dtype=np.int64)
  
  def size(self):
    return self._payoffs.shape[0]
  
  """ Payoff Retrieval """
  def get_subset_payoffs(self, fill_nan=None, *, sid: int=None, other_sids):
    payoffs = self.get_payoffs(fill_nan, sid=sid)
    payoffs = payoffs[other_sids]
    return payoffs

  def get_payoffs(self, fill_nan=None, *, sid: int=None):
    payoffs = self._payoffs.copy()
    if fill_nan is not None:
      assert isinstance(fill_nan, (int, float)), fill_nan
      payoffs[np.isnan(payoffs)] = fill_nan
    if sid is not None:
      payoffs = payoffs[sid]
    return payoffs

  def get_subset_counts(self, *, sid: int=None, other_sids):
    counts = self.get_counts(sid=sid)
    counts = counts[other_sids]
    return counts

  def get_counts(self, *, sid: int=None):
    counts = self._counts.copy()
    if sid is not None:
      counts = counts[sid]
    return counts

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    if from_scratch:
      self._payoffs = np.zeros([0] * 2, dtype=np.float32) * np.nan
      self._counts = np.zeros([0] * 2, dtype=np.int64) 
    else:
      self._payoffs = np.zeros_like(self._payoffs) * np.nan
      self._counts = np.zeros_like(self._counts)
    if name is not None:
      self.name = name

  def expand(self):
    pad_width = (0, 1)
    self._expand(pad_width)

  def update(self, sids: Tuple[int], scores: List[float]):
    assert len(sids) == 2, f'Strategies for both sides of agents were expected, but got {sids}'
    s_sum = sum(scores)
    s_total = len(scores)
    if sids[0] == sids[1]:
      self._payoffs[sids] = 0
    else:
      rsids = (sids[1], sids[0])
      if scores == []:
        return
      elif self._counts[sids] == 0:
        payoff = s_sum / s_total
        self._payoffs[sids] = payoff
        self._payoffs[rsids] = -payoff
      elif self.step_size == 0 or self.step_size is None:
        payoff = (self._counts[sids] * self._payoffs[sids] + s_sum) / (self._counts[sids] + s_total)
        self._payoffs[sids] = payoff
        self._payoffs[rsids] = -payoff
      else:
        payoff = s_sum / s_total
        new_payoff = self.step_size * (payoff - self._payoffs[sids])
        self._payoffs[sids] += new_payoff
        new_payoff = self.step_size * (-payoff - self._payoffs[rsids])
        self._payoffs[rsids] += new_payoff
      # assert self._payoffs[sids] + self._payoffs[rsids] == 1, (sids, self._payoffs[sids], rsids, self._payoffs[rsids])
      assert not np.isnan(self._payoffs[sids]), (self._counts[sids], self._payoffs[sids])
      self._counts[rsids] += s_total
    self._counts[sids] += s_total

  """ implementations """
  def _expand(self, pad_width):
    self._payoffs = np.pad(self._payoffs, pad_width, constant_values=np.nan)
    self._counts = np.pad(self._counts, pad_width)


class SelfPlayPayoffTableWithModel(SelfPlayPayoffTable):
  def __init__(self, step_size, dir, name='payoff'):
    # mappings between ModelPath and strategy index
    self._model2sid: Dict[ModelPath, int] = {}
    self._sid2model: List[ModelPath] = []

    super().__init__(step_size, dir, name)

  def __contains__(self, model: ModelPath):
    return model in self._sid2model

  """ Get & Set """
  def get_all_models(self):
    return self._sid2model

  def get_sid2model(self):
    return self._sid2model
  
  def get_model2sid(self):
    return self._model2sid

  """ Payoff Retrieval """
  def get_subset_payoffs(self, fill_nan=None, *, sid: int=None, model: ModelPath=None, 
                         other_sids: List[int]=None, other_models: List[ModelPath]=None):
    assert sid is not None or model is not None, "Must provide sid or model"
    assert other_sids is not None or other_models is not None, "Must provide other_sids or other_models"
    payoffs = self.get_payoffs(fill_nan, sid=sid, model=model)
    if other_sids is None and other_models is not None:
      other_sids = [self._model2sid[m] for m in other_models]
    payoffs = payoffs[other_sids]
    return payoffs

  def get_payoffs(self, fill_nan=None, *, sid: int=None, model: ModelPath=None):
    if sid is None and model is not None:
      sid = self._model2sid[model]
    payoff = super().get_payoffs(fill_nan=fill_nan, sid=sid)
    return payoff

  def get_subset_counts(self, *, sid: int=None, model: ModelPath=None, 
                        other_sids: List[int]=None, other_models: List[ModelPath]=None):
    assert sid is not None or model is not None, "Must provide sid or model"
    assert other_sids is not None or other_models is not None, "Must provide other_sids or other_models"
    counts = self.get_counts(sid=sid, model=model)
    if other_sids is None and other_models is not None:
      other_sids = [self._model2sid[m] for m in other_models]
    counts = counts[other_sids]
    return counts

  def get_counts(self, *, sid: int=None, model: ModelPath=None):
    if sid is None and model is not None:
      sid = self._model2sid[model]
    counts = super().get_counts(sid=sid)
    return counts

  """ Payoff Management """
  def reset(self, from_scratch=False, name=None):
    super().reset(from_scratch, name=name)
    if from_scratch:
      self._model2sid: Dict[ModelPath, int] = {}
      self._sid2model: List[ModelPath] = []

  def expand(self, model: ModelPath):
    self._expand_mappings(model)
    super().expand()
    self._check_consistency(model)

  def update(self, models: List[ModelPath], scores: List[float]):
    assert len(models) == 2, models
    sids = tuple([self._model2sid[model] for model in models if model])
    super().update(sids, scores)
    # print('Payoffs', *self._payoffs, 'Counts', *self._counts, sep='\n')

  """ Implementations """
  def _expand_mappings(self, model: ModelPath):
    assert isinstance(model, ModelPath), model
    assert model not in self._model2sid, f'Model({model}) is already in {list(self._model2sid)}'
    sid = self._payoffs.shape[0]
    self._model2sid[model] = sid
    self._sid2model.append(model)
    assert len(self._sid2model) == sid+1, (sid, self._sid2model)

  def _check_consistency(self, model: ModelPath):
    assert self._payoffs.shape[0] == self._model2sid[model] + 1, \
      (self._payoffs.shape[0], self._model2sid[model] + 1)
    assert self._counts.shape[0] == self._model2sid[model] + 1, \
      (self._counts.shape[0], self._model2sid[model] + 1)
