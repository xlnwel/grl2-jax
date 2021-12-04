import os
import random
import cloudpickle

from core.elements.builder import ElementsBuilder
from distributed.remote.base import RayBase
from run.utils import search_for_config
from utility.utils import config_attr, dict2AttrDict


class ParameterServer(RayBase):
    def __init__(self, config, env_stats, first_strategy_path='logs/card_gd/zero/self-play'):
        super().__init__()
        self.config = config_attr(self, config)
        self.env_stats = dict2AttrDict(env_stats)
        self.path = f'{self.config.root_dir}/{self.config.model_name}/parameter_server.pkl'
        if not os.path.exists(self.path) and not os.path.exists(first_strategy_path):
            raise ValueError('No strategy exists in the parameter server')
        # path to agent
        self._strategies = {}
        self.restore(first_strategy_path)

    def add_strategy_from_path(self, path, to_save=True):
        if path not in self._strategies:
            config = search_for_config(path)
            name = config.name
            builder = ElementsBuilder(config, self.env_stats, name=name)
            elements = builder.build_strategy_from_scratch(build_monitor=False)
            self._strategies[path] = elements.agent
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

    def restore(self, first_strategy_path):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                paths = cloudpickle.load(f)
                for p in paths:
                    self.add_strategy_from_path(p, to_save=False)
        else:
            self.add_strategy_from_path(first_strategy_path)
