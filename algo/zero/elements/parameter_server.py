import collections
import os
import random
import cloudpickle

from core.elements.builder import ElementsBuilder
from distributed.remote.base import RayBase
from env.func import get_env_stats
from run.utils import search_for_config
from utility.utils import config_attr, dict2AttrDict


class ParameterServer(RayBase):
    def __init__(self, config, env_stats):
        super().__init__()
        self.config = config_attr(self, config)
        self.env_stats = dict2AttrDict(env_stats)
        self.path = f'{self.config.root_dir}/{self.config.model_name}/parameter_server.pkl'
        # path to agent
        self._strategies = {}
        self._payoffs = collections.defaultdict(lambda: collections.defaultdict(float))
        self.restore()

    def is_empty(self):
        return len(self._strategies) == 0

    def add_strategy(self, path, weights):
        if path not in self._strategies:
            config = search_for_config(path)
            env_stats = get_env_stats(config.env)
            builder = ElementsBuilder(config, env_stats)
            elements = builder.build_strategy_from_scratch(build_monitor=False)
            self._strategies[path] = elements.strategy
            self.save()
        self._strategies[path].set_weights(weights)

    def add_strategy_from_path(self, path, to_save=True):
        if path not in self._strategies:
            config = search_for_config(path)
            env_stats = get_env_stats(config.env)
            builder = ElementsBuilder(config, env_stats)
            elements = builder.build_strategy_from_scratch(build_monitor=False)
            elements.strategy.restore(skip_trainer=True)
            self._strategies[path] = elements.strategy
        if to_save:
            self.save()

    def sample_strategy_path(self, k=1):
        if k == 1:
            return random.choice(list(self._strategies))
        else:
            return random.choices(list(self._strategies), k=k)

    def sample_strategy(self, k=1, opt_weights=False, actor_weights=False):
        if k == 1:
            strategy = random.choice(list(self._strategies.values()))
            weights = strategy.get_weights(
                opt_weights=opt_weights, 
                actor_weights=actor_weights)
        else:
            strategies = random.choices(list(self._strategies.values()), k=k)
            weights = [s.get_weights(
                opt_weights=opt_weights, 
                actor_weights=actor_weights) 
                for s in strategies]
        return weights

    def save(self):
         with open(self.path, 'wb') as f:
            cloudpickle.dump(list(self._strategies), f)

    def restore(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                paths = cloudpickle.load(f)
                for p in paths:
                    self.add_strategy_from_path(p, to_save=False)

