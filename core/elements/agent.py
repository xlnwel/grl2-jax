from core.decorator import *
from core.elements.strategy import Strategy
from core.monitor import Monitor
from core.typing import ModelPath
from utility.typing import AttrDict


class Agent:
    """ Initialization """
    def __init__(
        self, 
        *, 
        config: AttrDict,
        strategy: Strategy=None,
        monitor: Monitor=None,
        name: str=None,
        to_restore=True
    ):
        self.config = config
        self._name = name
        self._model_path = ModelPath(config.root_dir, config.model_name)
        self.strategies = {'default': strategy}
        self.strategy = strategy
        self.monitor = monitor

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

    def set_strategy(self, sid):
        self.strategy = self.strategies[sid]

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
        self.store(**stats)

    def save(self, print_terminal_info=False):
        for s in self.strategies.values():
            s.save(print_terminal_info=print_terminal_info)
    
    def restore(self):
        for s in self.strategies.values():
            s.restore()


def create_agent(**kwargs):
    return Agent(**kwargs)
