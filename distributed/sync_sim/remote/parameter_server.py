import collections
import itertools
import os
import random
from typing import Dict, List
import cloudpickle
import numpy as np
import ray

from ..common.typing import ModelWeights
from core.elements.builder import ElementsBuilderVC
from core.mixin.actor import RMSStats, combine_rms_stats
from core.remote.base import RayBase
from core.typing import ModelPath, get_aid, get_aid_vid
from distributed.sync_sim.remote.payoff import PayoffManager
from run.utils import search_for_config
from utility.display import pwt
from utility.utils import config_attr, dict2AttrDict
from utility import yaml_op


payoff = collections.defaultdict(lambda: collections.deque(maxlen=1000))
score = collections.defaultdict(lambda: 0)


""" Name Conventions:

We use "model" and "strategy" interchangeably. 
In general, we prefer the term "strategy" in the context of 
training and inference, and the term "model" when a model 
path is involved (e.g., when saving&restoring a model).
"""


""" Parameter Queue Management """
def _divide_runners(n_agents, n_runners, online_frac):
    n_self_play_runners = int(n_runners * online_frac)
    n_rest_queues = n_runners - n_self_play_runners
    n_agent_runners = n_rest_queues // n_agents
    assert n_agent_runners * n_agents + n_self_play_runners == n_runners, \
        (n_agent_runners, n_agents, n_self_play_runners, n_runners)

    return n_self_play_runners, n_agent_runners


class ParameterServer(RayBase):
    def __init__(
        self, 
        config: dict,  
        to_restore_params=True, 
        name='parameter_server',
    ):
        super().__init__(seed=config.get('seed'))
        self.config = dict2AttrDict(config['parameter_server'])
        self.name = name

        self.n_agents = config['n_agents']
        self.n_runners = config['runner']['n_runners']

        # the probability of training an agent from scratch
        self.train_from_scratch_frac = self.config.get('train_from_scratch_frac', 1)
        # self-play fraction
        self.online_frac = self.config.get('online_frac', .2)

        model_name = config["model_name"].rsplit('/')[0]
        self._dir = f'{config["root_dir"]}/{model_name}'
        os.makedirs(self._dir, exist_ok=True)
        self._path = f'{self._dir}/{self.name}.yaml'

        self.payoff_manager: PayoffManager = PayoffManager(
            self.config.payoff,
            self.n_agents, 
            self._dir,
        )
        self._params: List[Dict[ModelPath, Dict]] = [{} for _ in range(self.n_agents)]
        self._prepared_strategies: List[List[ModelWeights]] = \
            [[None for _ in range(self.n_agents)] for _ in range(self.n_runners)]
        self._ready = [False for _ in range(self.n_runners)]

        # an active model is the one under training
        self._active_models: List[ModelPath] = [None for _ in range(self.n_agents)]
        # is the first pbt iteration
        self._first_iteration = True
        self._all_strategies = None

        self.n_self_play_runners, self.n_agent_runners = \
            _divide_runners(
                self.n_agents, 
                self.n_runners, 
                self.online_frac, 
            )
        self.restore(to_restore_params)

    def build(
        self, 
        configs: List[dict], 
        env_stats: dict
    ):
        self.configs = [dict2AttrDict(c) for c in configs]
        self.builders: List[ElementsBuilderVC] = []
        for aid, config in enumerate(configs):
            model = f'{config["root_dir"]}/{config["model_name"]}'
            assert model.rsplit('/')[-1] == f'a{aid}', model
            os.makedirs(model, exist_ok=True)
            builder = ElementsBuilderVC(
                config, 
                env_stats, 
                to_save_code=False
            )
            self.builders.append(builder)
        assert len(self.builders) == self.n_agents, \
            (len(configs), len(self.builders), self.n_agents)

    def get_configs(self):
        return self.configs

    def get_active_models(self):
        return self._active_models

    """ Strategy Management """
    def get_strategies(self, rid: int=-1):
        if rid < 0:
            if not all(self._ready):
                return None
            strategies = self._prepared_strategies
            self._prepared_strategies = [
                [None for _ in range(self.n_agents)] 
                for _ in range(self.n_runners)
            ]
            self._ready = [False] *self.n_runners
            return strategies
        else:
            if not self._ready[rid]:
                return None
            strategies = self._prepared_strategies[rid]
            self._prepared_strategies[rid] = [None for _ in range(self.n_agents)]
            self._ready[rid] = False
            return strategies

    def update_strategy_weights(self, aid, model_weights: ModelWeights):
        def get_model_ids(aid, model: ModelPath, mid):
            assert aid == get_aid(model.model_name), (aid, model)
            models = self.payoff_manager.sample_strategies(aid, model)
            assert model in models, (model, models)
            assert len(models) == self.n_agents, (self.n_agents, models)
            # We directly store mid for the agent with aid below rather than storing the associated data
            weights = [{} for _ in range(self.n_agents)]
            for i, m in enumerate(models):
                if i == aid:
                    weights[i] = mid
                else:
                    for k in ['model', 'train_step', 'aux']:
                        weights[i][k] = self._params[i][m][k]
            model_weights_list = [
                mid if w == mid else ModelWeights(m, w)
                for m, w in zip(models, weights)
            ]
            mids = [mid if mw == mid else ray.put(mw) 
                for mw in model_weights_list]

            return mids
        
        def prepare_models(aid, model_weights: ModelWeights):
            model_weights.weights.pop('opt')
            model_weights.weights['aux'] = \
                self._params[aid][model_weights.model].get('aux', RMSStats({}, None))
            mid = ray.put(model_weights)

            if self._first_iteration or self.online_frac == 1:
                for rid in range(self.n_runners):
                    self._prepared_strategies[rid][aid] = mid
                    self._ready[rid] = all(
                        [m is not None for m in self._prepared_strategies[rid]])
            else:
                mids = get_model_ids(aid, model_weights.model, mid)
                rid_min = self.n_self_play_runners + aid * self.n_agent_runners
                rid_max = self.n_self_play_runners + (aid + 1) * self.n_agent_runners
                for rid in range(self.n_runners):
                    if rid < self.n_self_play_runners:
                        self._prepared_strategies[rid][aid] = mid
                        self._ready[rid] = all(
                            [m is not None for m in self._prepared_strategies[rid]])
                    elif rid_min <= rid < rid_max:
                        self._prepared_strategies[rid] = mids
                        self._ready[rid] = True

        assert self._active_models[aid] == model_weights.model, (self._active_models, model_weights.model)
        assert set(model_weights.weights) == set(['model', 'opt', 'train_step']), list(model_weights.weights)
        assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
        self._params[aid][model_weights.model].update(model_weights.weights)

        prepare_models(aid, model_weights)

    def update_strategy_aux_stats(self, aid, model_weights: ModelWeights):
        assert len(model_weights.weights) == 1, list(model_weights.weights)
        assert 'aux' in model_weights.weights, list(model_weights.weights)
        assert aid == get_aid(model_weights.model.model_name), (aid, model_weights.model)
        if self._params[aid][model_weights.model] is not None \
                and 'aux' in self._params[aid][model_weights.model]:
            self._params[aid][model_weights.model]['aux'] = combine_rms_stats(
                self._params[aid][model_weights.model]['aux'], 
                model_weights.weights['aux'],
            )
        else:
            self._params[aid][model_weights.model]['aux'] = model_weights.weights['aux']

    def sample_training_strategies(self):
        strategies = []
        is_raw_strategy = [False for _ in range(self.n_agents)]
        if any([am is not None for am in self._active_models]):
            # restore active strategies 
            for aid, model in enumerate(self._active_models):
                assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
                weights = self._params[aid][model].copy()
                weights.pop('aux', None)
                strategies.append(ModelWeights(model, weights))
                pwt('Restore active strategy', model)
                [b.save_config() for b in self.builders]
        else:
            assert all([am is None for am in self._active_models]), self._active_models
            for aid in range(self.n_agents):
                if self._first_iteration or random.random() < self.train_from_scratch_frac:
                    self.builders[aid].increase_version()
                    model = self.builders[aid].get_model_path()
                    assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
                    assert model not in self._params[aid], (model, list(self._params[aid]))
                    self._params[aid][model] = {}
                    weights = None
                    is_raw_strategy[aid] = True
                    pwt('A raw strategy is sampled for training:', model)
                else:
                    model = random.choice(list(self._params[aid]))
                    pwt(f'Sampling a historical stratgy from {list(self._params[aid])}')
                    assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
                    weights = self._params[aid][model].copy()
                    weights.pop('aux')
                    config = search_for_config(model)
                    model, config = self.builders[aid].get_sub_version(config)
                    assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}({model})'
                    assert model not in self._params[aid], f'{model} is already in {list(self._params[aid])}'
                    self._params[aid][model] = weights
                    pwt('A historical strategy is sampled for training:', model, config.version)
                self._active_models[aid] = model
                strategies.append(ModelWeights(model, weights))
            self.payoff_manager.add_strategies([s.model for s in strategies])
            self.save()

        return strategies, is_raw_strategy

    def sample_strategies_for_evaluation(self):
        if self._all_strategies is None:
            strategies = self.payoff_manager.get_all_strategies()
            self._all_strategies = [mw for mw in itertools.product(
                *[[s for s in ss] for ss in strategies])]
            assert len(self._all_strategies) == np.product([len(s) for s in strategies]), \
                (len(self._all_strategies), np.product([len(s) for s in strategies]))

        return self._all_strategies

    def archive_training_strategies(self):
        pwt('Archive training strategies')
        for model_path in self._active_models:
            self.save_params(model_path)
        self._active_models = [None for _ in range(self.n_agents)]
        self._first_iteration = False
        self.save()

    """ Payoff Operations """
    def reset_payoffs(self, from_scratch=True):
        self.payoff_manager.reset(from_scratch=from_scratch)

    def get_payoffs(self):
        return self.payoff_manager.get_payoffs()

    def get_counts(self):
        return self.payoff_manager.get_counts()

    def update_payoffs(self, models: List[ModelPath], scores: List[List[float]]):
        self.payoff_manager.update_payoffs(models, scores)

    """ Checkpoints """
    def save_active_model(self, model: ModelPath, train_step, env_step):
        pwt('Save active model', model)
        if model not in self._active_models:
            raise ValueError(f'{model} does not in active models{self._active_models}')
        aid = get_aid(model.model_name)
        if model not in self._params[aid]:
            raise ValueError(f'{model} does not in {list(self._params[aid])}')
        self._params[aid][model]['train_step'] = train_step
        self._params[aid][model]['env_step'] = env_step
        self.save_params(model)

    def save_params(self, model: ModelPath, filename='params'):
        assert model in self._active_models, (model, self._active_models)
        aid, vid = get_aid_vid(model.model_name)
        ps_dir = f'{self._dir}/a{aid}'
        if not os.path.isdir(ps_dir):
            os.makedirs(ps_dir)
        path = f'{ps_dir}/v{vid}/{filename}.pkl'
        with open(path, 'wb') as f:
            cloudpickle.dump(self._params[aid][model], f)
        pwt(f'Save parameters in "{path}"')

    def restore_params(self, model: ModelPath, filename='params'):
        aid, vid = get_aid_vid(model.model_name)
        path = f'{self._dir}/a{aid}/v{vid}/{filename}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._params[aid][model] = cloudpickle.load(f)
            pwt(f'Restore parameters from "{path}"')
        else:
            self._params[aid][model] = {}

    def save(self):
        self.payoff_manager.save()
        with open(self._path, 'wb') as f:
            model_paths = [[list(mn) for mn in p] for p in self._params]
            active_models = [list(m) if m is not None else m for m in self._active_models]
            yaml_op.dump(
                self._path, 
                model_paths=model_paths, 
                active_models=active_models, 
                first_iteration=self._first_iteration
            )

    def restore(self, to_restore_params=True):
        self.payoff_manager.restore()
        if os.path.exists(self._path):
            config = yaml_op.load(self._path)
            if config is None:
                return
            model_paths = config.pop('model_paths')
            self._active_models = [
                m if m is None else ModelPath(*m) 
                for m in config.pop('active_models')]
            config_attr(self, config, config_as_attr=False, private_attr=True)
            if to_restore_params:
                for models in model_paths:
                    for m in models:
                        self.restore_params(ModelPath(*m))

    """ Data Retrieval """
    def get_aux_stats(self, model_path: ModelPath):
        def rms2dict(rms: RMSStats):
            stats = {}
            if rms.obs:
                for k, v in rms.obs.items():
                    for kk, vv in v._asdict().items():
                        stats[f'aux/{k}/{kk}'] = vv
            if rms.reward:
                for k, v in rms.reward._asdict().items():
                    stats[f'aux/reward/{k}'] = v

            return stats

        aid = get_aid(model_path.model_name)
        rms = self._params[aid][model_path].get('aux', RMSStats({}, None))
        stats = rms2dict(rms)

        return stats


if __name__ == '__main__':
    from env.func import get_env_stats
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/builtin.yaml')
    env_stats = get_env_stats(config['env'])
    ps = ParameterServer(config, env_stats)
