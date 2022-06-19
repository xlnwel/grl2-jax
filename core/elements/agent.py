from typing import Dict, Union

from core.decorator import *
from core.elements.builder import ElementsBuilder
from core.elements.strategy import Strategy
from core.log import do_logging
from core.monitor import Monitor
from core.typing import ModelPath, get_algo
from distributed.sync.common.typing import ModelWeights
from run.utils import search_for_config
from utility.typing import AttrDict


class Agent:
    """ Initialization """
    def __init__(
        self, 
        *, 
        config: AttrDict,
        strategy: Union[Dict[str, Strategy], Strategy]=None,
        monitor: Monitor=None,
        name: str=None,
        to_restore=True, 
        builder: ElementsBuilder=None
    ):
        self.config = config
        self._name = name
        self._model_path = ModelPath(config.root_dir, config.model_name)
        if isinstance(strategy, dict):
            self.strategies: Dict[str, Strategy] = strategy
        else:
            self.strategies: Dict[str, Strategy] = {'default': strategy}
        self.strategy: Strategy = self.strategies[
            get_algo(self._model_path)
        ]
        self.monitor: Monitor = monitor
        self.builder: ElementsBuilder = builder
        # trainable is set to align with the first strategy
        self.is_trainable = self.strategy.is_trainable

        if to_restore:
            self.restore()

    @property
    def name(self):
        return self._name

    def reset_model_path(self, model_path: ModelPath):
        self.strategy.reset_model_path(model_path)
        if self.monitor:
            self.monitor.reset_model_path(model_path)

    def get_model_path(self):
        return self._model_path

    def add_strategy(self, sid, strategy: Strategy):
        self.strategies[sid] = strategy

    def switch_strategy(self, sid):
        self.strategy = self.strategies[sid]

    def set_strategy(self, strategy: ModelWeights, *, env=None):
        """
            strategy: strategy is rule-based if the model_name is int, 
            in which case strategy.weights is the config for that strategy 
            initialization. Otherwise, strategy is expected to be 
            learned by RL
        """
        self._model_path = strategy.model
        if len(strategy.model.root_dir.split('/')) < 3:
            # the strategy is rule-based if model_name is int(standing for version)
            # for rule-based strategies, we expect strategy.weights 
            # to be the kwargs for the strategy initialization
            algo = strategy.model
            if algo not in self.strategies:
                self.strategies[algo] = \
                    self.builder.build_rule_based_strategy(
                        env, 
                        strategy.weights
                    )
            self.monitor.reset_model_path(None)
        else:
            algo = get_algo(strategy.model)
            if algo not in self.strategies:
                config = search_for_config('/'.join(strategy.model))
                self.config = config
                build_func = self.builder.build_training_strategy_from_scratch \
                    if self.is_trainable else self.builder.build_acting_strategy_from_scratch
                elements = build_func(
                    config=config, 
                    env_stats=self.strategy.env_stats, 
                    build_monitor=self.monitor is not None
                )
                self.strategies[algo] = elements.strategy
                do_logging(f'Adding new strategy {strategy.model}', 
                    level='print', time=True)
            if self.monitor is not None and self.monitor.save_to_disk:
                self.monitor = self.monitor.reset_model_path(strategy.model)
            self.strategies[algo].set_weights(strategy.weights)

        self.strategy = self.strategies[algo]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        if hasattr(self.strategy, name):
            # Expose the interface of strategy as Agent and Strategy are interchangeably in many cases 
            return getattr(self.strategy, name)
        elif hasattr(self.monitor, name):
            return getattr(self.monitor, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def __call__(self, *args, **kwargs):
        return self.strategy(*args, **kwargs)

    """ Train """
    def train_record(self):
        stats = self.strategy.train_record()
        self.monitor.store(**stats)

    def save(self, print_terminal_info=False):
        for s in self.strategies.values():
            s.save(print_terminal_info=print_terminal_info)
    
    def restore(self):
        for s in self.strategies.values():
            s.restore()


def create_agent(**kwargs):
    return Agent(**kwargs)
