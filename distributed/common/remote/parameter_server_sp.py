import os
import copy
import filelock
import random
import cloudpickle
from functools import partial
from typing import Dict, List
import numpy as np
import ray

from core.ckpt import pickle
from core.elements.builder import ElementsBuilderVC
from core.log import do_logging
from core.mixin.actor import RMSStats, combine_rms_stats, rms2dict
from core.names import *
from core.remote.base import RayBase
from core.typing import AttrDict, AttrDict2dict, ModelPath, ModelWeights, \
  construct_model_name, exclude_subdict, get_aid, get_date, \
    get_basic_model_name, decompose_model_name, get_iid_vid
from rule.utils import is_rule_strategy
from run.utils import search_for_config
from tools.schedule import PiecewiseSchedule
from tools.timer import timeit
from tools.utils import config_attr, dict2AttrDict
from tools import yaml_op
from distributed.common.names import *
from distributed.common.typing import *
from distributed.common.remote.payoff import PayoffManager
from distributed.common.utils import *


""" Name Conventions:

We use "model" and "strategy" interchangeably. 
In general, we prefer the term "strategy" in the context of 
training and inference, and the term "model" when a model 
path is involved (e.g., when saving&restoring a model).
"""


timeit = partial(timeit, period=1)


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
    self.score_metrics = self.config.get('score_metrics', ScoreMetrics.SCORE)
    assert self.score_metrics in (ScoreMetrics.SCORE, ScoreMetrics.WIN_RATE)
    self.name = name

    self.n_agents = config.n_agents
    self.n_active_agents = 1
    self.n_runners = config.runner.n_runners

    # the probability of training an agent from scratch
    self.train_from_scratch_frac = self.config.get('train_from_scratch_frac', 1)
    self.train_from_latest_frac = self.config.get('train_from_latest_frac', 0)
    self.hist_train_scale = self.config.get('hist_train_scale', 1)
    self.reset_policy_head_frac = self.config.get('reset_policy_head_frac', 1)
    # fraction of runners devoted to the play of the most recent strategies 
    self.online_frac = self.config.get('online_frac', .2)
    self.online_scheduler = PiecewiseSchedule(self.online_frac, interpolation='stage')
    self.self_play = self.config.get('self_play', True)
    assert self.self_play, self.self_play

    model_name = get_basic_model_name(config.model_name)
    self._dir = os.path.join(config.root_dir, model_name)
    os.makedirs(self._dir, exist_ok=True)
    self._path = os.path.join(self._dir, f'{self.name}.yaml')

    self._pool_pattern = self.config.get('pool_pattern', '.*')
    self._pool_name = self.config.get('pool_name', 'strategy_pool')
    date = get_date(config.model_name)
    self._pool_dir = os.path.join(config.root_dir, f'{date}')
    self._pool_path = os.path.join(self._pool_dir, f'{self._pool_name}.yaml')

    self._params: Dict[ModelPath, Dict] = {}
    self._prepared_strategies: List[List[ModelWeights]] = \
      [[None for _ in range(2)] for _ in range(self.n_runners)]
    self._reset_ready()

    self._rule_strategies = set()

    self._opp_dist: Dict[ModelPath, List[float]] = {}

    self._models: Dict[str, ModelPath] = AttrDict()
    self._models[ModelType.FORMER] = None
    # an active model is the one under training
    self._models[ModelType.ACTIVE] = None
    self._models[ModelType.HARD] = {}
    self.hard_frac = self.config.get('hard_frac', .2)
    self.hard_threshold = self.config.get('hard_threshold', 0.5)
    self.n_hard_opponents = self.config.get('n_hard_opponents', 5)
    # is the first pbt iteration
    self._iteration = 1
    self._all_strategies = None

    self.payoff_manager: PayoffManager = PayoffManager(
      self.config.payoff, self.n_agents, self._dir, self_play=self.self_play)

    succ = self.restore(to_restore_params)
    self._update_runner_distribution()
    self.load_strategy_pool()

    if self.config.get('rule_strategies'):
      self.add_rule_strategies(self.config.rule_strategies, local=succ)

    self.check()

  def _reset_ready(self):
    self._ready = [False for _ in range(self.n_runners)]

  def check(self):
    assert self.payoff_manager.size() == len(self._params), (f'{self.payoff_manager.get_all_models()}\n{list(self._params)}')

  def build(self, configs: List[Dict], env_stats: Dict):
    self.agent_config = dict2AttrDict(configs[0])
    model = os.path.join(self.agent_config["root_dir"], self.agent_config["model_name"])
    os.makedirs(model, exist_ok=True)
    self.builder = ElementsBuilderVC(self.agent_config, env_stats, to_save_code=False)

  """ Strategy Pool Management """
  def save_strategy_pool(self):
    lock = filelock.FileLock(self._pool_path + '.lock')

    while True:
      with lock.acquire(timeout=5):
        config = yaml_op.load(self._pool_path)
        config[self._dir] = [list(m) for m in self._params]
        yaml_op.dump(self._pool_path, **config)
        break
      do_logging(f'{self._pool_path} is blocked for 5s', 'red')
    do_logging(f'Saving strategy pool to {self._pool_path}', color='green')

  @timeit
  def load_strategy_pool(self, to_search=True, pattern=None):
    """ Load strategy pool
    Args:
      to_search: if True, search the storage system to find all strategies;
      else, only load strategies defined in self._pool_path
    """
    if to_search:
      pattern = pattern or self._pool_pattern
      models = find_all_models(self._pool_dir, pattern)
      for model in models:
        if model not in self._params or self._params[model][STATUS] == Status.TRAINING:
          self.restore_params(model)
        if model not in self.payoff_manager:
          self.add_strategy_to_payoff(model)
      do_logging(f'Loading strategy pool in path {self._pool_dir}', color='green')
    else:
      lock = filelock.FileLock(self._pool_path + '.lock')

      while True:
        with lock.acquire(timeout=5):
          config = yaml_op.load(self._pool_path)
          break
        do_logging(f'{self._pool_path} is blocked for 5s', 'red')

      for v in config.values():
        for model in v:
          model = ModelPath(*model)
          if model not in self._params or self._params[model][STATUS] == Status.TRAINING:
            self.restore_params(model)
          if model not in self.payoff_manager:
            self.add_strategy_to_payoff(model)
      do_logging(f'Loading strategy pool from {self._pool_path}', color='green')
    self.save()
    self.check()

  """ Data Retrieval """
  def get_active_models(self):
    return [self._models[ModelType.ACTIVE]]

  def get_active_aux_stats(self):
    active_stats = {self._models[ModelType.ACTIVE]: self.get_aux_stats(self._models[ModelType.ACTIVE])}

    return active_stats

  def get_aux_stats(self, model_path: ModelPath):
    rms = self._params[model_path].get(ANCILLARY, RMSStats({}, None))
    stats = rms2dict(rms)

    return stats

  def get_opponent_distributions_for_active_models(self):
    payoff, _, dist = self.payoff_manager.compute_opponent_distribution(
      0, self._models[ModelType.ACTIVE], False)
    
    for x in dist:
      if x.size > 1:
        online_frac = self.online_scheduler(self._iteration)
        x /= np.nansum(x[:-1]) / (1 - online_frac)
        x[-1] = online_frac
    dists = {self._models[ModelType.ACTIVE]: (payoff, dist)}

    return dists

  def get_runner_stats(self):
    self._update_runner_distribution()
    if self._iteration == 1 and not self._rule_strategies:
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
      model_name = os.path.join(model_name, f'{name}-rule')
      model_name = construct_model_name(model_name, aid, vid, vid)
      model = ModelPath(self.config.root_dir, model_name)
      self._rule_strategies.add(model)
      self._params[model] = AttrDict2dict(config)
      models.append(model)
      do_logging(f'Adding rule strategy: {model}', color='green')
      if not local:
        # Add the rule strategy to the payoff table if the payoff manager is not restored from a checkpoint
        self.payoff_manager.add_strategy(model)

  def add_strategy_to_payoff(self, model: ModelPath):
    self.payoff_manager.add_strategy(model)

  def _update_active_model(self, model: ModelPath):
    self._models[ModelType.ACTIVE] = model

  def _reset_prepared_strategy(self, rid: int=-1):
    raise NotImplementedError

  def get_prepared_strategies(self, rid: int=-1):
    if rid < 0:
      if not all(self._ready):
        return None
      strategies = self._prepared_strategies
    else:
      if not self._ready[rid]:
        return None
      strategies = self._prepared_strategies[rid]
    self._reset_prepared_strategy(rid)
    return strategies

  def update_and_prepare_strategy(
    self, 
    aid: int, 
    model_weights: ModelWeights, 
    step=None
  ):
    assert aid == 0, aid
    assert self._models[ModelType.ACTIVE] == model_weights.model, (self._models[ModelType.ACTIVE], model_weights.model)
    assert set(model_weights.weights) == set([MODEL, OPTIMIZER, TRAIN_STEP]), list(model_weights.weights)
    assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
    
    self._params[model_weights.model].update(model_weights.weights)
    # self.save_params(model_weights.model, to_print=False)
    model_weights = ModelWeights(model_weights.model, model_weights.weights.copy())
    self._prepare_models(model_weights)
    assert all(self._ready), self._ready

  def _put_model_weights(self, model):
    if model in self._rule_strategies:
      # rule-based strategy
      weights = self._params[model]
    else:
      # if error happens here
      # it's likely that you retrive the latest model 
      # in self.payoff_manager.sample_opponent_strategies
      weights = {
        k: self._params[model][k] 
        for k in [MODEL, TRAIN_STEP, ANCILLARY]
        if k in self._params[model]
      }
    mid = ray.put(ModelWeights(model, weights))
    return mid

  def _get_historical_mids(self, mid, model):
    opp_model = self.sample_opponent_strategies(model)
    opp_mid = self._put_model_weights(opp_model)
    mids = [mid, opp_mid]

    return mids
  
  def _prepare_recent_models(self, mid):
    # prepare the most recent model for the first n_runners runners
    for rid in range(self.n_online_runners):
      self._prepared_strategies[rid] = [mid, mid]
      self._ready[rid] = True

  def _prepare_historical_models(self, mid, model):
    mids = self._get_historical_mids(mid, model)
    assert len(mids) == self.n_agents, (len(mids), self.n_agents)
    for rid in range(self.n_online_runners, self.n_runners):
      self._prepared_strategies[rid] = mids
      self._ready[rid] = True

  def _prepare_models(self, model_weights: ModelWeights):
    model_weights.weights.pop(OPTIMIZER)
    model_weights.weights[ANCILLARY] = \
      self._params[model_weights.model].get(ANCILLARY, RMSStats([], None))
    mid = ray.put(model_weights)

    # prepare the most recent models for online runners
    self._prepare_recent_models(mid)
    if self.n_online_runners < self.n_runners:
      # prepare historical models for selected runners
      self._prepare_historical_models(mid, model_weights.model)

  def update_aux_stats(self, aid, model_weights: ModelWeights):
    assert aid == 0, aid
    assert len(model_weights.weights) == 1, list(model_weights.weights)
    assert ANCILLARY in model_weights.weights, list(model_weights.weights)
    assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
    if self._params[model_weights.model] is not None \
        and ANCILLARY in self._params[model_weights.model]:
      self._params[model_weights.model][ANCILLARY] = combine_rms_stats(
        self._params[model_weights.model][ANCILLARY], 
        model_weights.weights[ANCILLARY],
      )
    else:
      self._params[model_weights.model][ANCILLARY] = model_weights.weights[ANCILLARY]

  def _update_runner_distribution(self):
    if self._iteration == 1 and not self._rule_strategies:
      self.n_online_runners = self.n_runners
      self.n_agent_runners = 0
    else:
      online_frac = self.online_scheduler(self._iteration)
      self.n_online_runners, self.n_agent_runners = divide_runners(
        self.n_active_agents, self.n_runners, online_frac
      )

  def _retrieve_strategy_for_training(
      self, model: ModelPath, config=None, 
      new_model: ModelPath=None, reset_heads=False):
    if new_model is None:
      new_model = model
    do_logging(f'Retrieving strategy {model} for {new_model}', color='green')
    self.restore_params(model)
    weights = self._params[model].copy()
    weights.pop(ANCILLARY, None)
    if reset_heads:
      weights = reset_policy_head(weights, config)
      self._params[new_model] = weights
    model_weights = ModelWeights(new_model, weights)
    self._params[new_model] = weights
    self._params[model][STATUS] = Status.TRAINING
    if config is None:
      config = self.builder.config.copy(shallow=False)
    config.status = Status.TRAINING
    self.builder.save_config(config)
    return model_weights

  def _construct_raw_strategy(self, iteration, version=None):
    self.builder.set_iteration_version(iteration, version)
    config = self.builder.config.copy(shallow=False)
    config.status = Status.TRAINING
    self.builder.save_config(config)
    model = self.builder.get_model_path()
    assert model not in self._params, (model, list(self._params))
    self._params[model] = {}
    self._params[model][STATUS] = Status.TRAINING
    weights = None
    model_weights = ModelWeights(model, weights)
    do_logging(f'Sampling raw strategy for training: {model}', color='green')
    
    return model_weights

  def _sample_with_prioritization(self):
    candidates = []
    candidate_scores = []
    candidate_weights = []
    for m in self._params:
      if not is_rule_strategy(m):
        candidates.append(m)
        score = self.get_avg_score(0, m)
        candidate_scores.append(score)
    candidate_scores = np.array(candidate_scores)
    candidate_scores = candidate_scores - np.min(candidate_scores) + .01 # avoid zero weights
    candidate_weights = candidate_scores ** self.hist_train_scale
    idx = random.choices(range(len(candidates)), candidate_weights)[0]
    model = candidates[idx]
    model_scores = [f'{m}={s}' for m, s in zip(self._params, candidate_scores)]
    do_logging(f'Sampling historical stratgy({model}={candidate_scores[idx]})'
                f' from {model_scores}', color='green')
    return model

  def _sample_historical_strategy(self, iteration):
    if self._models[ModelType.FORMER] and random.random() < self.train_from_latest_frac:
      model = self._models[ModelType.FORMER]
      do_logging(f'Sampling historical stratgy({model}) from {list(self._params)}', color='green')
    else:
      model = self._sample_with_prioritization()
    config = search_for_config(model)
    new_model = self.builder.get_sub_version(config, iteration)
    assert new_model not in self._params, f'{new_model} is already in {list(self._params)}'
    reset_heads = random.random() < self.reset_policy_head_frac
    model_weights = self._retrieve_strategy_for_training(
      model, config, new_model, reset_heads=reset_heads)
    
    return model_weights

  def sample_training_strategies(self, iteration=None):
    if iteration is not None:
      assert iteration == self._iteration, (iteration, self._iteration)
    strategies = []
    is_raw_strategy = [False for _ in range(self.n_active_agents)]
    if self._models[ModelType.ACTIVE] is not None:
      model_weights = self._retrieve_strategy_for_training(self._models[ModelType.ACTIVE])
      strategies.append(model_weights)
    else:
      assert self._models[ModelType.ACTIVE] is None, self._models[ModelType.ACTIVE]
      if self._iteration == 1 or random.random() < self.train_from_scratch_frac:
        model_weights = self._construct_raw_strategy(self._iteration)
        is_raw_strategy[0] = True
      else:
        model_weights = self._sample_historical_strategy(self._iteration)
      strategies.append(model_weights)
      model = strategies[0].model
      self.add_strategy_to_payoff(model)
      self._update_active_model(model)
      self.save_active_models(0, 0)
      self.save_strategy_pool()
      self.save()

    return strategies, is_raw_strategy

  def archive_training_strategies(self, status: Status):
    do_logging(f'Archiving training strategies: {self._models[ModelType.ACTIVE]}', color='green')
    config = self.builder.config.copy(shallow=False)
    config.status = status
    self.builder.save_config(config)
    self._models[ModelType.FORMER] = copy.copy(self._models[ModelType.ACTIVE])
    self._params[self._models[ModelType.FORMER]][STATUS] = status
    self.save_params(self._models[ModelType.ACTIVE])
    self._update_active_model(None)
    if self._models[ModelType.ACTIVE] in self._opp_dist:
      del self._opp_dist[self._models[ModelType.ACTIVE]]
    self._iteration += 1
    self._reset_ready()
    self._update_runner_distribution()
    self.save()

  """ Strategy Sampling """
  def sample_opponent_strategies(self, model: ModelPath):
    payoffs, weights, opp_dist = self._compute_opp_distributions(model)
    self._update_hard_opponents(model, payoffs, opp_dist)
    if len(self._models[ModelType.HARD][model]) > 0 \
        and random.random() < self.hard_frac:
      opp_model = random.choice(self._models[ModelType.HARD][model])
    else:
      sid2model = self.payoff_manager.get_sid2model()
      opp_model = random.choices(sid2model[:-1], weights=opp_dist[:-1])[0]
    return opp_model

  def sample_strategies_for_evaluation(self):
    if self._all_strategies is None:
      strategies = self.payoff_manager.get_all_strategies()
      self._all_strategies = []
      for i, s in enumerate(strategies):
        for j in range(i+1, len(strategies)):
          self._all_strategies.append([s, strategies[j]])
    return self._all_strategies

  """ Payoff Operations """
  def get_avg_score(self, aid: int, model: ModelPath):
    payoffs = self.get_payoffs_for_model(aid, model)
    n_valid_payoffs = np.sum(~np.isnan(payoffs))
    if n_valid_payoffs > payoffs.size / 2:
      return np.nanmean(payoffs)
    else:
      if self.score_metrics == ScoreMetrics.SCORE:
        return -1
      else:
        return 0

  def reset_payoffs(self, from_scratch=True, name=None):
    self.payoff_manager.reset(from_scratch=from_scratch, name=name)

  def get_payoffs(self, fill_nan=False):
    return self.payoff_manager.get_payoffs(fill_nan=fill_nan)

  def get_payoffs_for_model(self, aid: int, model: ModelPath):
    return self.payoff_manager.get_payoffs_for_model(aid, model)

  def get_counts(self):
    return self.payoff_manager.get_counts()

  def update_payoffs(self, models: List[ModelPath], scores: List[List[float]]):
    self.payoff_manager.update_payoffs(models, scores)
    self.payoff_manager.save(to_print=False)

  def _compute_opp_distributions(self, model: ModelPath):
    assert isinstance(model, ModelPath), model
    payoffs, weights, opp_dists = self.payoff_manager.\
      compute_opponent_distribution(0, model)
    return payoffs[0], weights[0], opp_dists[0]

  def _update_opp_distributions(self, model: ModelPath):
    assert isinstance(model, ModelPath), model
    payoffs, weights, opp_dist = self._compute_opp_distributions(model)
    self._opp_dist[model] = [opp_dist]
    do_logging(f'Updating opponent distributions ({opp_dist}) '
               f'with weights ({weights}) and payoffs ({payoffs})', color='green')
    return payoffs, weights, opp_dist

  def _update_hard_opponents(self, model: ModelPath, payoffs, dist):
    sid2model = self.payoff_manager.get_sid2model()
    sids = np.argsort(dist)[-self.n_hard_opponents:][::-1]
    sids = [sid for sid in sids if payoffs[sid] > 0]
    self._models[ModelType.HARD][model] = [sid2model[sid] for sid in sids]
    return self._models[ModelType.HARD][model]

  """ Checkpoints """
  def save_active_model(self, model: ModelPath, train_step: int=None, 
                        env_step: int=None, to_print=True):
    if to_print:
      do_logging(f'Saving active model: {model}', color='green')
    assert model == self._models[ModelType.ACTIVE], (model, self._models[ModelType.ACTIVE])
    assert model in self._params, f'{model} does not in {list(self._params)}'
    if train_step is not None:
      self._params[model][TRAIN_STEP] = train_step
    if env_step is not None:
      self._params[model][ENV_STEP] = env_step
    self.save_params(model, to_print=to_print)

  def save_active_models(self, train_step: int=None, env_step: int=None, to_print=True):
    self.save_active_model(self._models[ModelType.ACTIVE], train_step, 
                           env_step, to_print=to_print)

  def save_params(self, model: ModelPath, name='params', to_print=True):
    assert model == self._models[ModelType.ACTIVE], (model, self._models[ModelType.ACTIVE])
    if MODEL in self._params[model]:
      pickle.save_params(self._params[model][MODEL], 
                         model, f'{name}/model', to_print=to_print)
    if OPTIMIZER in self._params[model]:
      pickle.save_params(self._params[model][OPTIMIZER], 
                         model, f'{name}/opt', to_print=to_print)
    rest_params = exclude_subdict(self._params[model], MODEL, OPTIMIZER)
    if rest_params:
      pickle.save_params(rest_params, model, name, to_print=to_print)

  def restore_params(self, model: ModelPath, name='params'):
    self._params[model] = pickle.restore_params(model, name, backtrack=6)

  def save(self):
    self.payoff_manager.save()
    with open(self._path, 'wb') as f:
      cloudpickle.dump((
        self._params, 
        self._models, 
        self._iteration, 
        self.n_online_runners, 
        self.n_agent_runners
      ), f)
    # model_paths = [list(mn) for mn in self._params]
    # models = jax.tree_map(
    #   lambda x: None if x is None else list(x), self._models, 
    #   is_leaf=lambda x: isinstance(x, ModelPath) or x is None)
    # yaml_op.dump(
    #   self._path, 
    #   model_paths=model_paths, 
    #   models=models, 
    #   iteration=self._iteration, 
    #   n_online_runners=self.n_online_runners, 
    #   n_agent_runners=self.n_agent_runners
    # )

  def restore(self, to_restore_params=True):
    self.payoff_manager.restore()
    if os.path.exists(self._path):
      with open(self._path, 'rb') as f:
        self._params, \
        self._models, \
        self._iteration, \
        self.n_online_runners, \
        self.n_agent_runners = cloudpickle.load(f)
    #   config = yaml_op.load(self._path)
    #   if config is None:
    #     return
    #   models = config.pop('models')
    #   models = jax.tree_map(
    #     lambda x: ModelPath(*x), models, 
    #     is_leaf=lambda x: isinstance(x, list) or x is None)
    #   self._models = models

    #   model_paths = config.pop('model_paths')
    #   if to_restore_params:
    #     for model in model_paths:
    #       model = ModelPath(*model)
    #       if not is_rule_strategy(model):
    #         self.restore_params(model)
      
    #   config_attr(self, config, config_as_attr=False, private_attr=True)
      return True
    else:
      return False


class ExploiterSPParameterServer(SPParameterServer):
  def __init__(
    self, 
    config, 
    to_restore_params=True, 
    name='parameter_server'
  ):
    super().__init__(config, to_restore_params, name=name)
    self._models[ModelType.Target] = None
    self.online_frac = self.config.get('exploiter_online_frac', 0.2)
    self.online_scheduler = PiecewiseSchedule(self.online_frac, interpolation='stage')
    self.target_frac = self.config.get('target_frac', 0.7)

  def sample_training_strategies(self, iteration=None):
    if iteration is not None:
      assert iteration == self._iteration, (iteration, self._iteration)
    target_dir = self._dir.replace(EXPLOITER_SUFFIX, '')
    target_dir = os.path.join(target_dir, 'a0')
    target_model = find_latest_model(target_dir)
    self.restore_params(target_model)
    if target_model not in self.payoff_manager:
      self.add_strategy_to_payoff(target_model)

    model = self._get_exploiter_model(target_model)
    assert model.model_name.replace(EXPLOITER_SUFFIX, '') == target_model.model_name, (model_name, target_model.model_name)
    iid, vid = get_iid_vid(model.model_name)
    self.builder.set_iteration_version(iid, vid)
    if model in self._params:
      model_weights = self._retrieve_strategy_for_training(model)
      is_raw_strategy = [False]
    else:
      if self._iteration == 1 or random.random() < self.train_from_scratch_frac:
        model_weights = self._construct_raw_strategy(iid, vid)
        assert model == model_weights.model, (model, model_weights.model)
      else:
        payoffs, _, _ = self._compute_opp_distributions(target_model)
        sid2model = self.payoff_manager.get_sid2model()
        sid = np.argmin(payoffs)
        hardest_model = sid2model[sid]
        self.restore_params(hardest_model)
        reset_heads = random.random() < self.reset_policy_head_frac
        model_weights = self._retrieve_strategy_for_training(
          hardest_model, new_model=model, reset_heads=reset_heads)
      is_raw_strategy = [True]
    strategies = [model_weights]
    self.add_strategy_to_payoff(model)
    self._update_active_model(model)
    self.save_active_models(0, 0)
    self.save_strategy_pool()
    self.save()
    return strategies, is_raw_strategy

  def sample_opponent_strategies(self, model: ModelPath):
    if random.random() < self.target_frac:
      opp_model = self._get_exploitee_model(model)
      if opp_model not in self._params:
        self.restore_params(opp_model)
    else:
      _, _, opp_dist = self._compute_opp_distributions(model)
      sid2model = self.payoff_manager.get_sid2model()
      opp_model = random.choices(sid2model[:-1], weights=opp_dist[:-1])[0]
    return opp_model
  
  def _get_exploiter_model(self, model: ModelPath):
    assert EXPLOITER_SUFFIX not in model.model_name, model.model_name
    basic_name, aid, iid, vid = decompose_model_name(model.model_name)
    basic_name, seed = basic_name.rsplit(PATH_SPLIT, maxsplit=1)
    basic_name = basic_name + EXPLOITER_SUFFIX
    basic_name = os.path.join(basic_name, seed)
    model_name = construct_model_name(basic_name, aid, iid, vid)
    return ModelPath(model.root_dir, model_name)

  def _get_exploitee_model(self, model: ModelPath):
    assert EXPLOITER_SUFFIX in model.model_name, model.model_name
    return ModelPath(model.root_dir, model.model_name.replace(EXPLOITER_SUFFIX, ''))

  def get_avg_score(self, aid: int, model: ModelPath):
    if self.score_metrics == ScoreMetrics.SCORE:
      return -1
    else:
      return 0


if __name__ == '__main__':
  from env.func import get_env_stats
  from tools.yaml_op import load_config
  config = load_config('algo/gd/configs/builtin.yaml')
  ps = SPParameterServer(config)
