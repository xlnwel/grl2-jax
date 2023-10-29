import collections
import itertools
import os
import random
from typing import Dict, List
import numpy as np
import ray

from core.ckpt import pickle
from core.elements.builder import ElementsBuilderVC
from core.log import do_logging
from core.mixin.actor import RMSStats, combine_rms_stats, rms2dict
from core.remote.base import RayBase
from core.typing import AttrDict, AttrDict2dict, ModelPath, construct_model_name, exclude_subdict, \
  get_aid, get_basic_model_name
from core.typing import ModelWeights
from distributed.sync.remote.payoff import PayoffManager
from rule.utils import is_rule_strategy
from run.utils import search_for_config
from tools.schedule import PiecewiseSchedule
from tools.timer import Every
from tools.utils import config_attr, dict2AttrDict
from tools import yaml_op


""" Name Conventions:

We use "model" and "strategy" interchangeably. 
In general, we prefer the term "strategy" in the context of 
training and inference, and the term "model" when a model 
path is involved (e.g., when saving&restoring a model).
"""


def _divide_runners(n_agents, n_runners, online_frac):
  if n_runners < n_agents:
    return n_runners, 0
  n_agent_runners = int(n_runners * (1 - online_frac) // n_agents)
  n_online_runners = n_runners - n_agent_runners * n_agents
  assert n_agent_runners * n_agents + n_online_runners == n_runners, \
    (n_agent_runners, n_agents, n_online_runners, n_runners)
  return n_online_runners, n_agent_runners


class SPParameterServer(RayBase):
  def __init__(
    self, 
    config: dict,  
    to_restore_params=True, 
    name='parameter_server',
  ):
    super().__init__(seed=config.get('seed'))
    config = dict2AttrDict(config)
    self.config = config.parameter_server
    self.name = name

    self.n_agents = config.n_agents
    self.n_active_agents = 1
    self.n_runners = config.runner.n_runners

    # the probability of training an agent from scratch
    self.train_from_scratch_frac = self.config.get('train_from_scratch_frac', 1)
    # fraction of runners devoted to the play of the most recent strategies 
    self.online_frac = self.config.get('online_frac', .2)
    self.online_scheduler = PiecewiseSchedule(self.online_frac, interpolation='stage')
    self.self_play = self.config.get('self_play', True)
    assert self.self_play, self.self_play

    model_name = get_basic_model_name(config.model_name)
    self._dir = f'{config.root_dir}/{model_name}'
    os.makedirs(self._dir, exist_ok=True)
    self._path = f'{self._dir}/{self.name}.yaml'

    self._params: Dict[ModelPath, Dict] = {}
    self._prepared_strategies: List[List[ModelWeights]] = \
      [[None for _ in range(2)] for _ in range(self.n_runners)]
    self._reset_ready()

    self._rule_strategies = set()

    self._opp_dist: Dict[ModelPath, List[float]] = {}
    self._to_update: Dict[ModelPath, Every] = collections.defaultdict(
      lambda: Every(self.config.setdefault('update_interval', 1), -1))

    # an active model is the one under training
    self._active_model: ModelPath = None
    # is the first pbt iteration
    self._iteration = 1
    self._all_strategies = None

    self.payoff_manager: PayoffManager = PayoffManager(
      self.config.payoff, self.n_agents, self._dir, self_play=self.self_play)

    succ = self.restore(to_restore_params)
    self._update_runner_distribution()

    if self.config.get('rule_strategies'):
      self.add_rule_strategies(self.config.rule_strategies, local=succ)

    self.check()

  def _reset_ready(self):
    self._ready = [False for _ in range(self.n_runners)]

  def check(self):
    pass

  def build(self, configs: List[Dict], env_stats: Dict):
    self.agent_config = dict2AttrDict(configs[0])
    model = f'{self.agent_config["root_dir"]}/{self.agent_config["model_name"]}'
    os.makedirs(model, exist_ok=True)
    self.builder = ElementsBuilderVC(self.agent_config, env_stats, to_save_code=False)

  """ Data Retrieval """
  def get_active_models(self):
    return [self._active_model]

  def get_active_aux_stats(self):
    active_stats = {self._active_model: self.get_aux_stats(self._active_model)}

    return active_stats

  def get_aux_stats(self, model_path: ModelPath):
    rms = self._params[model_path].get('aux', RMSStats({}, None))
    stats = rms2dict(rms)

    return stats

  def get_opponent_distributions_for_active_models(self):
    payoff, dist = self.payoff_manager.get_opponent_distribution(
      0, self._active_model, False)
    
    for x in dist:
      if x.size > 1:
        online_frac = self.online_scheduler(self._iteration)
        x /= np.nansum(x[:-1]) / (1 - online_frac)
        x[-1] = online_frac
    dists = {self._active_model: (payoff, dist)}

    return dists

  def get_runner_stats(self):
    self._update_runner_distribution()
    if self._iteration == 1:
      stats = AttrDict(
        iteration=self._iteration, 
        online_frac=1,
        n_online_runners=self.n_online_runners, 
        n_agent_runners=self.n_agent_runners, 
      )
    else:
      stats = AttrDict(
        iteration=self._iteration, 
        online_frac=self.online_scheduler(self._iteration), 
        n_online_runners=self.n_online_runners, 
        n_agent_runners=self.n_agent_runners, 
      )

    return stats

  """ Strategy Management """
  def add_rule_strategies(self, rule_config: dict, local=False):
    models = []
    for name, config in rule_config.items():
      aid = config['aid']
      assert aid < self.n_active_agents, (aid, self.n_active_agents)
      vid = config['vid']
      model_name = get_basic_model_name(self.config.model_name)
      model_name = f'{model_name}/{name}-rule'
      model_name = construct_model_name(model_name, aid, vid, vid)
      model = ModelPath(self.config.root_dir, model_name)
      self._rule_strategies.add(model)
      self._params[model] = AttrDict2dict(config)
      models.append(model)
      do_logging(f'Adding rule strategy {model}')
      if not local:
        do_logging(f'Adding rule strategy to payoff table')
        self.payoff_manager.add_strategy(model)

  def _add_strategy_to_payoff(self, models: ModelPath):
    self.payoff_manager.add_strategy(models)

  def _update_active_model(self, model: ModelPath):
    self._active_model = model

  def get_strategies(self, rid: int=-1):
    if rid < 0:
      if not all(self._ready):
        return None
      strategies = self._prepared_strategies
      self._prepared_strategies = [
        [None for _ in range(self.n_agents)] 
        for _ in range(self.n_runners)
      ]
      self._ready = [False] * self.n_runners
    else:
      if not self._ready[rid]:
        return None
      strategies = self._prepared_strategies[rid]
      self._prepared_strategies[rid] = [None for _ in range(self.n_agents)]
      self._ready[rid] = False
    return strategies

  def update_and_prepare_strategy(
    self, 
    aid: int, 
    model_weights: ModelWeights, 
    step=None
  ):
    def put_model_weights(model):
      if model in self._rule_strategies:
        # rule-based strategy
        weights = self._params[model]
      else:
        # if error happens here
        # it's likely that you retrive the latest model 
        # in self.payoff_manager.sample_strategies
        weights = {k: self._params[model][k] 
                   for k in ['model', 'train_step', 'aux']}
      mid = ray.put(ModelWeights(model, weights))
      return mid

    def get_model_ids(mid, model):
      opp_model = self.sample_strategies_with_opp_dists(step, model)
      opp_mid = put_model_weights(opp_model)
      mids = [mid, opp_mid]

      return mids
    
    def prepare_recent_models(mid):
       # prepare the most recent model for the first n_runners runners
      for rid in range(self.n_online_runners):
        self._prepared_strategies[rid] = [mid, mid]
        self._ready[rid] = True

    def prepare_historical_models(mid, model):
      mids = get_model_ids(mid, model)
      assert len(mids) == self.n_agents, (len(mids), self.n_agents)
      for rid in range(self.n_online_runners, self.n_runners):
        self._prepared_strategies[rid] = mids
        self._ready[rid] = True

    def prepare_models(model_weights: ModelWeights):
      model_weights.weights.pop('opt')
      model_weights.weights['aux'] = \
        self._params[model_weights.model].get('aux', RMSStats([], None))
      mid = ray.put(model_weights)

      if self._iteration == 1 or self.n_runners == self.n_online_runners:
        # prepare the most recent model for all runners
        prepare_recent_models(mid)
      else:
        # prepare the most recent model for online runners
        prepare_recent_models(mid)
        
        # prepare historical models for selected runners
        prepare_historical_models(mid, model_weights.model)

    assert aid == 0, aid
    assert self._active_model == model_weights.model, (self._active_model, model_weights.model)
    assert set(model_weights.weights) == set(['model', 'opt', 'train_step']), list(model_weights.weights)
    assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
    
    self._params[model_weights.model].update(model_weights.weights)
    model_weights = ModelWeights(model_weights.model, model_weights.weights.copy())
    prepare_models(model_weights)
    assert all(self._ready), self._ready

  def update_aux_stats(self, aid, model_weights: ModelWeights):
    assert aid == 0, aid
    assert len(model_weights.weights) == 1, list(model_weights.weights)
    assert 'aux' in model_weights.weights, list(model_weights.weights)
    assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
    if self._params[model_weights.model] is not None \
        and 'aux' in self._params[model_weights.model]:
      self._params[model_weights.model]['aux'] = combine_rms_stats(
        self._params[model_weights.model]['aux'], 
        model_weights.weights['aux'],
      )
    else:
      self._params[model_weights.model]['aux'] = model_weights.weights['aux']

  def sample_training_strategies(self, iteration=None):
    if iteration is not None:
      assert iteration == self._iteration, (iteration, self._iteration)
    strategies = []
    is_raw_strategy = [False for _ in range(self.n_active_agents)]
    if self._active_model is not None:
      strategies = self._restore_active_strategies()
    else:
      assert self._active_model is None, self._active_model
      if self._iteration == 1 or random.random() < self.train_from_scratch_frac:
        model_weights = self._construct_raw_strategy(self._iteration)
        is_raw_strategy[0] = True
      else:
        model_weights = self._sample_historical_strategy(self._iteration)
      strategies.append(model_weights)
      model = strategies[0].model
      self._add_strategy_to_payoff(model)
      self._update_active_model(model)
      self._save_active_model()
      self.save()

    return strategies, is_raw_strategy

  def _update_runner_distribution(self):
    if self._iteration == 1:
      self.n_online_runners = self.n_runners
      self.n_agent_runners = 0
    else:
      online_frac = self.online_scheduler(self._iteration)
      self.n_online_runners, self.n_agent_runners = _divide_runners(
        self.n_active_agents, self.n_runners, online_frac
      )

  def _restore_active_strategies(self):
    # restore active strategies 
    strategies = []
    model = self._active_model
    weights = self._params[model].copy()
    weights.pop('aux', None)
    strategies.append(ModelWeights(model, weights))
    do_logging(f'Restoring active strategy: {model}')
    self.builder.save_config()
    return strategies

  def _construct_raw_strategy(self, iteration):
    self.builder.set_iteration(iteration)
    model = self.builder.get_model_path()
    assert model not in self._params, (model, list(self._params))
    self._params[model] = {}
    weights = None
    model_weights = ModelWeights(model, weights)
    do_logging(f'Sampling raw strategy for training: {model}')
    
    return model_weights

  def _sample_historical_strategy(self, iteration):
    model = random.choice([m for m in self._params if not is_rule_strategy(m)])
    do_logging(f'Sampling historical stratgy({model}) from {list(self._params)}')
    weights = self._params[model].copy()
    weights.pop('aux')
    config = search_for_config(model)
    model, config = self.builder.get_sub_version(config, iteration)
    assert model not in self._params, f'{model} is already in {list(self._params)}'
    self._params[model] = weights
    model_weights = ModelWeights(model, weights)
    
    return model_weights
  
  def archive_training_strategies(self):
    do_logging('Archiving training strategies', level='pwt', color='blue')
    self.save_params(self._active_model)
    self._update_active_model(None)
    if self._active_model in self._opp_dist:
      del self._opp_dist[self._active_model]
    self._iteration += 1
    self._reset_ready()
    self._update_runner_distribution()
    self.save()

  """ Strategy Sampling """
  def sample_strategies_with_opp_dists(self, step, model: ModelPath):
    if step is None or self._to_update[model](step):
      self._update_opp_distributions(model)
    opp_dist = self._opp_dist[model]
    sid2model = self.payoff_manager.get_sid2model()
    for m in sid2model:
      assert isinstance(m, ModelPath), m
    model = random.choices(
      sid2model[:-1], 
      weights=opp_dist[0][:-1]
    )[0]
    return model

  def _update_opp_distributions(self, model: ModelPath):
    assert isinstance(model, ModelPath), model
    payoffs, self._opp_dist[model] = self.payoff_manager.\
      get_opponent_distribution(0, model)
    do_logging(f'Updating opponent distributions: {self._opp_dist[model]} with payoffs {payoffs}')

  def sample_strategies_for_evaluation(self):
    if self._all_strategies is None:
      strategies = self.payoff_manager.get_all_strategies()
      self._all_strategies = [mw for mw in itertools.product(
        *[[s for s in ss] for ss in strategies])]
      assert len(self._all_strategies) == np.product([len(s) for s in strategies]), \
        (len(self._all_strategies), np.product([len(s) for s in strategies]))

    return self._all_strategies

  """ Payoff Operations """
  def reset_payoffs(self, from_scratch=True, name=None):
    self.payoff_manager.reset(from_scratch=from_scratch, name=name)

  def get_payoffs(self, fill_nan=False):
    return self.payoff_manager.get_payoffs(fill_nan=fill_nan)

  def get_counts(self):
    return self.payoff_manager.get_counts()

  def update_payoffs(self, models: List[ModelPath], scores: List[List[float]]):
    self.payoff_manager.update_payoffs(models, scores)
    self.payoff_manager.save(to_print=False)

  """ Checkpoints """
  def save_active_model(self, model: ModelPath, train_step: int, env_step: int):
    do_logging(f'Saving active model: {model}')
    assert model == self._active_model, (model, self._active_model)
    assert model in self._params, f'{model} ot in {list(self._params)}'
    self._params[model]['train_step'] = train_step
    self._params[model]['env_step'] = env_step
    self.save_params(model)

  def _save_active_model(self):
    self.save_active_model(self._active_model, 0, 0)

  def save_params(self, model: ModelPath, name='params'):
    assert model == self._active_model, (model, self._active_model)
    if 'model' in self._params[model]:
      pickle.save_params(
        self._params[model]['model'], model, f'{name}/model')
    if 'opt' in self._params[model]:
      pickle.save_params(
        self._params[model]['opt'], model, f'{name}/opt')
    rest_params = exclude_subdict(self._params[model], 'model', 'opt')
    if rest_params:
      pickle.save_params(rest_params, model, name)

  def restore_params(self, model: ModelPath, name='params'):
    params = pickle.restore_params(model, name)
    self._params[model] = params

  def save(self):
    self.payoff_manager.save()
    model_paths = [list(mn) for mn in self._params]
    active_model = None if self._active_model is None else list(self._active_model)
    yaml_op.dump(
      self._path, 
      model_paths=model_paths, 
      active_model=active_model, 
      iteration=self._iteration, 
      n_online_runners=self.n_online_runners, 
      n_agent_runners=self.n_agent_runners
    )

  def restore(self, to_restore_params=True):
    self.payoff_manager.restore()
    if os.path.exists(self._path):
      config = yaml_op.load(self._path)
      if config is None:
        return
      self._active_model = ModelPath(*config.pop('active_model'))
      config_attr(self, config, config_as_attr=False, private_attr=True)
      model_paths = config.pop('model_paths')
      if to_restore_params:
        for model in model_paths:
          model = ModelPath(*model)
          if not is_rule_strategy(model):
            self.restore_params(model)
      return True
    else:
      return False


if __name__ == '__main__':
  from env.func import get_env_stats
  from tools.yaml_op import load_config
  config = load_config('algo/gd/configs/builtin.yaml')
  env_stats = get_env_stats(config['env'])
  ps = ParameterServer(config, env_stats)
