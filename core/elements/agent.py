from core.elements.strategy import Strategy
from core.decorator import *
from core.monitor import Monitor
from core.utils import save_code
from utility.utils import config_attr


class Agent:
    """ Initialization """
    def __init__(self, 
                 *, 
                 config: dict,
                 strategy: Strategy=None,
                 monitor: Monitor=None,
                 name=None):
        config_attr(self, config)
        self._name = name
        self.strategies = {'default': strategy}
        self.strategy = strategy
        self.monitor = monitor

        self.restore()
        save_code(self._root_dir, self._model_name)

    @property
    def name(self):
        return self._name

    def add_strategy(self, sid, strategy):
        self.strategies[sid] = strategy

    def set_strategy(self, sid):
        self.strategy = self.strategies[sid]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        if hasattr(self.strategy, name):
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
