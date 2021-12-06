import collections
import os
import random
import cloudpickle
import numpy as np

from core.elements.builder import ElementsBuilder
from core.remote.base import RayBase
from core.typing import Path
from env.func import get_env_stats
from run.utils import search_for_config
from utility.utils import dict2AttrDict


class ParameterServer(RayBase):
    def __init__(self, config, env_stats):
        super().__init__()
        self.config = dict2AttrDict(config['parameter_server'])
        self.p = self.config.p
        self.default_score = self.config.default_score
        self.env_stats = dict2AttrDict(env_stats)
        self.builder = ElementsBuilder(config, env_stats)

        path = f'{self.config.root_dir}/{self.config.model_name}'
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = f'{path}/parameter_server.pkl'
        # {main path: strategies}
        self._strategies = {}
        # {main path: {other path: [payoff]}}
        self._payoffs = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.deque(maxlen=1000)))
        # {main path: {other path: weights}}
        self._scores = collections.defaultdict(lambda: collections.defaultdict(float))
        self._latest_strategy_path = None
        self.restore()

    def is_empty(self):
        return len(self._strategies) == 0

    """ Strategy Operations """
    def add_strategy(self, root_dir, model_name, weights):
        path_tuple = Path(root_dir, model_name)
        path = '/'.join(path_tuple)
        if path_tuple not in self._strategies:
            config = search_for_config(path)
            elements = self.builder.build_actor_strategy_from_scratch(
                config, build_monitor=False)
            self._strategies[path_tuple] = elements.strategy
            self.save()
        self._strategies[path_tuple].set_weights(weights)
        self._latest_strategy_path = path_tuple

    def add_strategy_from_path(self, root_dir, model_name):
        path_tuple = Path(root_dir, model_name)
        path = '/'.join(path_tuple)
        if path_tuple not in self._strategies:
            config = search_for_config(path)
            elements = self.builder.build_actor_strategy_from_scratch(
                config, build_monitor=False)
            elements.strategy.restore(skip_trainer=True)
            self._strategies[path_tuple] = elements.strategy
            self.save()
        self._latest_strategy_path = path_tuple

    def sample_strategy_path(self, main_root_dir, main_model_name):
        scores = self.get_scores(main_root_dir, main_model_name)
        weights = self.get_weights_vector(scores)
        assert len(self._strategies) == len(weights), (len(self._strategies), len(weights))
        return random.choices(list(self._strategies), weights=weights)[0]

    def sample_strategy(self, main_root_dir, main_model_name):
        path = self.sample_strategy_path(main_root_dir, main_model_name)
        strategy = self._strategies[path]
        strategy_weights = strategy.get_weights()

        return path, strategy_weights

    def retrieve_latest_strategy_path(self):
        return self._latest_strategy_path

    def retrieve_latest_strategy_weights(self):
        return self._strategies[self._latest_strategy_path].get_weights()

    """ Payoffs/Weights Operations """
    def add_payoff(self, main_root_dir, main_model_name, other_root_dir, other_model_name, payoff):
        main_path = Path(main_root_dir, main_model_name)
        other_path = Path(other_root_dir, other_model_name)
        assert main_path in self._strategies, (main_path, list(self._strategies))
        assert other_path in self._strategies, (other_path, list(self._strategies))
        self._payoffs[main_path][other_path] += payoff
        self.compute_scores(main_path, other_path)

    def compute_scores(self, main_path, other_path):
        for k in self._strategies.keys():
            if k not in self._scores[main_path]:
                self._scores[main_path][k] = self.default_score
        self._scores[main_path][other_path] = np.mean(self._payoffs[main_path][other_path])

    def compute_weights(self, scores):
        weights = (1 - scores)**self.p
        return weights

    def get_scores(self, main_root_dir, main_model_name):
        main_path = Path(main_root_dir, main_model_name)
        scores = {k: self._scores[main_path][k] for k in self._strategies.keys()}
        return scores

    def get_weights_vector(self, scores):
        scores = np.array(list(scores.values()))
        weights = self.compute_weights(scores)
        return weights

    def get_scores_and_weights(self, main_root_dir, main_model_name):
        scores = self.get_scores(main_root_dir, main_model_name)
        weights = self.get_weights_vector(scores)
        weights = weights / np.sum(weights)
        weights = {f'{k.model_name}_weights': w for k, w in zip(scores.keys(), weights)}
        scores = {f'{k.model_name}_scores': v for k, v in scores.items()}
        return scores, weights                          

    """ Checkpoints """
    def save(self):
         with open(self.path, 'wb') as f:
            cloudpickle.dump((list(self._strategies), self._payoffs, self._scores), f)

    def restore(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                paths, self._payoffs, self._scores = cloudpickle.load(f)
                for p in paths:
                    self.add_strategy_from_path(*p)


if __name__ == '__main__':
    from env.func import get_env_stats
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/builtin.yaml')
    env_stats = get_env_stats(config['env'])
    ps = ParameterServer(config, env_stats)
