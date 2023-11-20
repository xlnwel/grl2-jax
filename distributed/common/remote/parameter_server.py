import collections
import itertools
import os
import random
from typing import Dict, List
import numpy as np
import jax
import ray

from core.ckpt import pickle
from core.elements.builder import ElementsBuilderVC
from core.log import do_logging
from core.mixin.actor import RMSStats, combine_rms_stats, rms2dict
from core.names import *
from core.remote.base import RayBase
from core.typing import AttrDict, AttrDict2dict, ModelPath, construct_model_name, exclude_subdict, \
  get_aid, get_basic_model_name
from core.typing import ModelWeights
from distributed.common.remote.payoff import PayoffManager
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


class ParameterServer(RayBase):
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
    self.n_runners = config.runner.n_runners

    # the probability of training an agent from scratch
    self.train_from_scratch_frac = self.config.get('train_from_scratch_frac', 1)
    self.train_from_latest_frac = self.config.get('train_from_latest_frac', 0)
    # fraction of runners devoted to the play of the most recent strategies 
    self.online_frac = self.config.get('online_frac', .2)
    self.online_scheduler = PiecewiseSchedule(self.online_frac, interpolation='stage')
    self.self_play = self.config.get('self_play', False)
    self._reset_policy_head = self.config.get('reset_policy_head', True)
    assert not self.self_play, self.self_play

    model_name = get_basic_model_name(config.model_name)
    self._dir = f'{config.root_dir}/{model_name}'
    os.makedirs(self._dir, exist_ok=True)
    self._path = f'{self._dir}/{self.name}.yaml'

    self._params: List[Dict[ModelPath, Dict]] = [{} for _ in range(self.n_agents)]
    self._prepared_strategies: List[List[ModelWeights]] = \
      [[None for _ in range(self.n_agents)] for _ in range(self.n_runners)]
    self._reset_ready()

    self._rule_strategies = set()

    self._opp_dist: Dict[ModelPath, List[float]] = {}
    self._to_update: Dict[ModelPath, Every] = collections.defaultdict(
      lambda: Every(self.config.setdefault('update_interval', 1), -1))

    self._former_models: List[ModelPath] = [None for _ in range(self.n_agents)]
    # an active model is the one under training
    self._active_models: List[ModelPath] = [None for _ in range(self.n_agents)]
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
    self.configs = [dict2AttrDict(c) for c in configs]
    self.builders: List[ElementsBuilderVC] = []
    for aid, config in enumerate(configs):
      model = f'{config["root_dir"]}/{config["model_name"]}'
      assert model.rsplit('/')[-1] == f'a{aid}', model
      os.makedirs(model, exist_ok=True)
      builder = ElementsBuilderVC(config, env_stats, to_save_code=False)
      self.builders.append(builder)
    assert len(self.builders) == self.n_agents, (len(configs), len(self.builders), self.n_agents)

  """ Data Retrieval """
  def get_configs(self):
    return self.configs

  def get_active_models(self):
    return self._active_models

  def get_active_aux_stats(self):
    active_stats = {m: self.get_aux_stats(m) for m in self._active_models}

    return active_stats

  def get_aux_stats(self, model_path: ModelPath):
    aid = get_aid(model_path.model_name)
    rms = self._params[aid][model_path].get(ANCILLARY, RMSStats({}, None))
    stats = rms2dict(rms)

    return stats

  def get_opponent_distributions_for_active_models(self):
    dists = {
      m: self.payoff_manager.get_opponent_distribution(i, m, False) 
      for i, m in enumerate(self._active_models)
    }
    for m, (p, d) in dists.items():
      for x in d:
        if x.size > 1:
          online_frac = self.online_scheduler(self._iteration)
          x /= np.nansum(x[:-1]) / (1 - online_frac)
          x[-1] = online_frac
      dists[m] = (p, d)

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
      assert aid < self.n_agents, (aid, self.n_agents)
      vid = config['vid']
      model_name = get_basic_model_name(self.config.model_name)
      model_name = f'{model_name}/{name}-rule'
      model_name = construct_model_name(model_name, aid, vid, vid)
      model = ModelPath(self.config.root_dir, model_name)
      self._rule_strategies.add(model)
      self._params[aid][model] = AttrDict2dict(config)
      models.append(model)
      do_logging(f'Adding rule strategy {model}', color='green')
      if not local:
        do_logging(f'Adding rule strategy to payoff table', color='green')
        self.payoff_manager.add_strategy(model, aid=aid)

  def add_strategies_to_payoff(self, models: List[ModelPath]):
    assert len(models) == self.n_agents, models
    self.payoff_manager.add_strategies(models)

  def _update_active_model(self, aid, model: ModelPath):
    self._active_models[aid] = model

  def _update_active_models(self, models: List[ModelPath]):
    assert len(models) == self.n_agents, models
    self._active_models = models

  def _reset_prepared_strategy(self, rid: int=-1):
    pass

  def get_strategies(self, rid: int=-1):
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
    def put_model_weights(aid, mid, models):
      mids = []
      for i, m in enumerate(models):
        if i == aid:
          # We directly store mid for the agent with aid
          mids.append(mid)
        else:
          if m in self._rule_strategies:
            # rule-based strategy
            weights = self._params[i][m]
          else:
            # if error happens here
            # it's likely that you retrive the latest model 
            # in self.payoff_manager.sample_strategies
            weights = {k: self._params[i][m][k] 
              for k in [MODEL, 'train_step', ANCILLARY]}
          mids.append(ray.put(ModelWeights(m, weights)))
      return mids

    def get_historical_mids(aid, mid, model_weights: ModelWeights):
      model = model_weights.model
      assert aid == get_aid(model.model_name), (aid, model)
      models = self.sample_strategies(aid, model, step)
      assert model in models, (model, models)
      assert len(models) == self.n_agents, (self.n_agents, models)
      mids = put_model_weights(aid, mid, models)
      assert len(mids) == self.n_agents, (self.n_agents, mids)

      return mids
    
    def prepare_recent_models(aid, mid, n_runners):
       # prepare the most recent model for the first n_runners runners
      for rid in range(n_runners):
        self._prepared_strategies[rid][aid] = mid
        self._ready[rid] = all(
          [m is not None for m in self._prepared_strategies[rid]]
        )

    def prepare_historical_models(aid, mid, model_weights: ModelWeights):
      rid_min = self.n_online_runners + aid * self.n_agent_runners
      rid_max = self.n_online_runners + (aid + 1) * self.n_agent_runners
      mids = get_historical_mids(aid, mid, model_weights)
      for rid in range(rid_min, rid_max):
        self._prepared_strategies[rid] = mids
        self._ready[rid] = True

    def prepare_models(aid, model_weights: ModelWeights):
      model_weights.weights.pop(OPTIMIZER)
      model_weights.weights[ANCILLARY] = \
        self._params[aid][model_weights.model].get(ANCILLARY, RMSStats({}, None))
      mid = ray.put(model_weights)

      if self._iteration == 1 or self.n_runners == self.n_online_runners:
        # prepare the most recent model for all runners
        prepare_recent_models(aid, mid, self.n_runners)
      else:
        # prepare the most recent model for online runners
        prepare_recent_models(aid, mid, self.n_online_runners)
        
        # prepare historical models for selected runners
        prepare_historical_models(aid, mid, model_weights)

    assert self._active_models[aid] == model_weights.model, (self._active_models, model_weights.model)
    assert set(model_weights.weights) == set([MODEL, OPTIMIZER, 'train_step']), list(model_weights.weights)
    assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
    
    self._params[aid][model_weights.model].update(model_weights.weights)
    model_weights = ModelWeights(model_weights.model, model_weights.weights.copy())
    prepare_models(aid, model_weights)

  def update_aux_stats(self, aid, model_weights: ModelWeights):
    assert len(model_weights.weights) == 1, list(model_weights.weights)
    assert ANCILLARY in model_weights.weights, list(model_weights.weights)
    assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
    if self._params[aid][model_weights.model] is not None \
        and ANCILLARY in self._params[aid][model_weights.model]:
      self._params[aid][model_weights.model][ANCILLARY] = combine_rms_stats(
        self._params[aid][model_weights.model][ANCILLARY], 
        model_weights.weights[ANCILLARY],
      )
    else:
      self._params[aid][model_weights.model][ANCILLARY] = model_weights.weights[ANCILLARY]

  def sample_training_strategies(self, iteration=None):
    if iteration is not None:
      assert iteration == self._iteration, (iteration, self._iteration)
    strategies = []
    is_raw_strategy = [False for _ in range(self.n_agents)]
    if any([am is not None for am in self._active_models]):
      strategies = self._restore_active_strategies()
    else:
      assert all([am is None for am in self._active_models]), self._active_models
      for aid in range(self.n_agents):
        if self._iteration == 1 or random.random() < self.train_from_scratch_frac:
          model_weights = self._construct_raw_strategy(aid, self._iteration)
          is_raw_strategy[aid] = True
        else:
          model_weights = self._sample_historical_strategy(aid, self._iteration)
        strategies.append(model_weights)
      models = [s.model for s in strategies]
      self.add_strategies_to_payoff(models)
      self._update_active_models(models)
      self._save_active_models()
      self.save()

    return strategies, is_raw_strategy

  def _update_runner_distribution(self):
    if self._iteration == 1:
      self.n_online_runners = self.n_runners
      self.n_agent_runners = 0
    else:
      online_frac = self.online_scheduler(self._iteration)
      self.n_online_runners, self.n_agent_runners = _divide_runners(
        self.n_agents, self.n_runners, online_frac
      )

  def _restore_active_strategies(self):
    # restore active strategies 
    strategies = []
    for aid, model in enumerate(self._active_models):
      assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
      weights = self._params[aid][model].copy()
      weights.pop(ANCILLARY, None)
      strategies.append(ModelWeights(model, weights))
      do_logging(f'Restoring active strategy: {model}', color='green')
      [b.save_config() for b in self.builders]
    return strategies

  def _construct_raw_strategy(self, aid, iteration):
    self.builders[aid].set_iteration(iteration)
    model = self.builders[aid].get_model_path()
    assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
    assert model not in self._params[aid], (model, list(self._params[aid]))
    self._params[aid][model] = {}
    weights = None
    model_weights = ModelWeights(model, weights)
    do_logging(f'Sampling raw strategy for training: {model}', color='green')
    
    return model_weights

  def _sample_historical_strategy(self, aid, iteration):
    candidates = [m for m in self._params[aid] if not is_rule_strategy(m)]
    if self._former_models[aid] and random.random() < self.train_from_latest_frac:
      model = self._former_models[aid]
    else:
      model = random.choice(candidates)
    do_logging(f'Sampling historical stratgy({model}) from {list(self._params[aid])}', color='green')
    assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
    weights = self._params[aid][model].copy()
    weights.pop(ANCILLARY)
    config = search_for_config(model)
    model, config = self.builders[aid].get_sub_version(config, iteration)
    assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
    assert model not in self._params[aid], f'{model} is already in {list(self._params[aid])}'
    if self._reset_policy_head:
      rng = jax.random.PRNGKey(random.randint(0, 2**32))
      out = weights[MODEL]['policy']['policy/mlp/out']
      w = jax.nn.initializers.orthogonal(.01)(rng, out['w'].shape)
      b = jax.nn.initializers.zeros(rng, out['b'].shape)
      out['w'] = w
      out['b'] = b
    self._params[aid][model] = weights
    model_weights = ModelWeights(model, weights)
    
    return model_weights
  
  def archive_training_strategies(self):
    do_logging('Archiving training strategies', color='green')
    self._former_models = self._active_models.copy()
    for aid, model in enumerate(self._active_models):
      self.save_params(model)
      self._update_active_model(aid, None)
      if model in self._opp_dist:
        del self._opp_dist[model]
    self._iteration += 1
    self._reset_ready()
    self._update_runner_distribution()
    self.save()

  """ Strategy Sampling """
  def sample_strategies(self, aid, model: ModelPath, step=None):
    assert model == self._active_models[aid], (model, self._active_models)
    if step is None or self._to_update[model](step):
      self._update_opp_distributions(aid, model)

    strategies = self.sample_strategies_with_opp_dists(
      aid, model, self._opp_dist[model]
    )
    assert model in strategies, (model, strategies)

    return strategies
  
  def sample_strategies_with_opp_dists(
    self, aid, model: ModelPath, opp_dists: List[np.ndarray],
  ):
    sid2model = self.payoff_manager.get_sid2model()
    models = [
      (random.choices(
        sid2model[i][:-1], 
        weights=opp_dists[i if i < aid else i-1][:-1]
      )[0] if len(sid2model[i]) > 1 else sid2model[i][0])
      if i != aid else model
      for i in range(self.n_agents)
    ]
    return models

  def _update_opp_distributions(self, aid, model: ModelPath):
    assert isinstance(model, ModelPath), model
    payoffs, self._opp_dist[model] = self.payoff_manager.\
      get_opponent_distribution(aid, model)
    do_logging(f'Updating opponent distributions for agent {aid}: {self._opp_dist[model]} with payoffs {payoffs}', color='green')

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
    do_logging(f'Saving active model: {model}', color='green')
    assert model in self._active_models, (model, self._active_models)
    aid = get_aid(model.model_name)
    assert model in self._params[aid], f'{model} does not in {list(self._params[aid])}'
    self._params[aid][model]['train_step'] = train_step
    self._params[aid][model]['env_step'] = env_step
    self.save_params(model)

  def _save_active_models(self):
    for m in self._active_models:
      self.save_active_model(m, 0, 0)

  def save_params(self, model: ModelPath, name='params'):
    assert model in self._active_models, (model, self._active_models)
    aid = get_aid(model.model_name)
    if MODEL in self._params[aid][model]:
      pickle.save_params(
        self._params[aid][model][MODEL], model, f'{name}/model')
    if OPTIMIZER in self._params[aid][model]:
      pickle.save_params(
        self._params[aid][model][OPTIMIZER], model, f'{name}/opt')
    rest_params = exclude_subdict(self._params[aid][model], MODEL, OPTIMIZER)
    if rest_params:
      pickle.save_params(rest_params, model, name)

  def restore_params(self, model: ModelPath, name='params'):
    aid = get_aid(model.model_name)
    params = pickle.restore_params(model, name)
    self._params[aid][model] = params

  def save(self):
    self.payoff_manager.save()
    model_paths = [[list(mn) for mn in p] for p in self._params]
    active_models = [list(m) if m is not None else m for m in self._active_models]
    yaml_op.dump(
      self._path, 
      model_paths=model_paths, 
      active_models=active_models, 
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
      model_paths = config.pop('model_paths')
      for aid, model in enumerate(config.pop('active_models')):
        if model is not None:
          model = ModelPath(*model)
        self._update_active_model(aid, model)
      config_attr(self, config, config_as_attr=False, private_attr=True)
      if to_restore_params:
        for models in model_paths:
          for m in models:
            m = ModelPath(*m)
            if not is_rule_strategy(m):
              self.restore_params(m)
      return True
    else:
      return False


if __name__ == '__main__':
  from env.func import get_env_stats
  from tools.yaml_op import load_config
  config = load_config('algo/gd/configs/builtin.yaml')
  env_stats = get_env_stats(config['env'])
  ps = ParameterServer(config, env_stats)
