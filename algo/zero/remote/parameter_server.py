import collections
import os
import random
import cloudpickle
import numpy as np

from core.elements.builder import ElementsBuilder
from core.remote.base import RayBase
from core.typing import ModelPath
from env.func import get_env_stats
from run.utils import search_for_all_configs, search_for_config
from utility.utils import dict2AttrDict


payoff = collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.deque(maxlen=1000)))
score = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))


class ParameterServer(RayBase):
    def __init__(self, config, env_stats, name='parameter_server'):
        super().__init__()
        self.config = dict2AttrDict(config['parameter_server'])
        self.p = self.config.p
        self.default_score = .5
        self.env_stats = dict2AttrDict(env_stats)
        self.builder = ElementsBuilder(config, env_stats)

        path = f'{self.config.root_dir}/{self.config.model_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        self._stats_path = f'{path}/{name}.pkl'
        # {main path: strategies}
        self._strategies = {}
        # {main path: {other path: [payoff]}}
        self._payoffs = payoff
        # {main path: {other path: weights}}
        self._scores = score
        self._latest_strategy_path = None
        self.restore()

    def is_empty(self, excluded_path):
        paths = [p for p in self._strategies.keys() if p != excluded_path]
        return len(paths) == 0

    def search_for_strategies(self, strategy_dir):
        configs = search_for_all_configs(strategy_dir)
        for c in configs:
            self.add_strategy_from_config(c)
    
    """ Strategy Operations """
    def add_strategy_from_config(self, config):
        model_path = ModelPath(config.root_dir, config.model_name)
        if model_path not in self._strategies:
            elements = self.builder.build_actor_strategy_from_scratch(
                config, build_monitor=False)
            elements.strategy.restore()
            self._strategies[model_path] = elements.strategy
            self.save()
        self._latest_strategy_path = model_path
        print('Strategies in the pool:', list(self._strategies))

    def add_strategy_from_path(self, model_path):
        path = os.path.join(*model_path)
        if model_path not in self._strategies:
            config = search_for_config(path)
            elements = self.builder.build_actor_strategy_from_scratch(
                config, build_monitor=False)
            elements.strategy.restore()
            self._strategies[model_path] = elements.strategy
            self.save()
        self._latest_strategy_path = model_path
        print('Strategies in the pool:', list(self._strategies))

    def update_strategy(self, model_path, weights):
        self._strategies[model_path].set_weights(weights)
        self._latest_strategy_path = model_path

    def set_latest_strategy(self, model_path):
        self._latest_strategy_path = model_path

    def sample_strategy_path(self, main_path):
        scores = self.get_scores(main_path)
        weights = self.get_weights_vector(scores)
        return random.choices([k for k in self._strategies if k != main_path], weights=weights)[0]

    def sample_strategy(self, main_path):
        path = self.sample_strategy_path(main_path)
        strategy = self._strategies[path]
        strategy_weights = strategy.get_weights()

        return path, strategy_weights

    def retrieve_latest_strategy_path(self):
        return self._latest_strategy_path

    def retrieve_latest_strategy_weights(self):
        return self._strategies[self._latest_strategy_path].get_weights()

    """ Payoffs/Weights Operations """
    def add_payoff(self, main_path, other_path, payoff):
        assert main_path in self._strategies, (main_path, list(self._strategies))
        assert other_path in self._strategies, (other_path, list(self._strategies))
        self._payoffs[main_path][other_path] += payoff

    def compute_scores(self, main_path, to_save=True):
        scores = self._scores[main_path] if to_save else {}
        for other_path in self._strategies.keys():
            if other_path != main_path:
                score = np.mean(self._payoffs[main_path][other_path]) \
                    if other_path in self._payoffs[main_path] else self.default_score
                scores[other_path] = score
                if score < self.default_score:
                    self.default_score = score
        return scores

    def compute_weights(self, scores):
        weights = (1 - scores)**self.p
        return weights

    def get_scores(self, main_path):
        scores = {k: self._scores[main_path][k] 
            for k in self._strategies.keys() if k != main_path}
        return scores

    def get_weights_vector(self, scores):
        scores = np.array(list(scores.values()))
        weights = self.compute_weights(scores)
        return weights

    def get_scores_and_weights(self, main_path):
        scores = {other_path: self._scores[main_path][other_path]
            for other_path in self._strategies.keys() if other_path != main_path}
        weights = self.get_weights_vector(scores)
        weights = weights / np.sum(weights)
        weights = {f'{k.model_name}_weights': v 
            for k, v in zip(scores.keys(), weights)}
        scores = {f'{k.model_name}_scores': v 
            for k, v in scores.items()}

        return scores, weights

    """ Checkpoints """
    def save(self):
        print(list(self._strategies), self._payoffs, self._scores)
        with open(self._stats_path, 'wb') as f:
            cloudpickle.dump((list(self._strategies), self._payoffs, self._scores), f)

    def restore(self):
        if os.path.exists(self._stats_path):
            with open(self._stats_path, 'rb') as f:
                paths, self._payoffs, self._scores = cloudpickle.load(f)
            for p in paths:
                self.add_strategy_from_path(p)


if __name__ == '__main__':
    from env.func import get_env_stats
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/builtin.yaml')
    env_stats = get_env_stats(config['env'])
    ps = ParameterServer(config, env_stats)
