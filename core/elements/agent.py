from core.elements.strategy import Strategy
from core.decorator import *
from core.dataset import create_dataset
from core.monitor import Monitor, create_monitor
from core.utils import save_code
from distributed.remote.base import RayBase
from utility.utils import config_attr, dict2AttrDict
from utility import pkg


class Agent:
    """ Initialization """
    def __init__(self, 
                 *, 
                 config: dict,
                 strategy: Strategy=None,
                 monitor: Monitor=None,
                 name=None,
                 to_save_code=True):
        self.config = config_attr(self, config)
        self.algorithm = self.config.algorithm
        self._name = name
        self.strategies = {'default': strategy}
        self.strategy = strategy
        self.monitor = monitor

        self.restore()
        if to_save_code:
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


# class RemoteAgent(RayBase):
#     def __init__(self,
#                  *,
#                  config,
#                  env_stats,
#                  create_model,
#                  create_loss,
#                  create_trainer,
#                  create_actor,
#                  create_buffer,
#                  create_strategy,
#                  create_agent,
#                  to_save_code=False):
#         super().__init__()
#         self.config = config = dict2AttrDict(config)
#         env_stats = dict2AttrDict(env_stats)
#         self._name = config.name

#         algo = self.config.algorithm
#         create_model = pkg.import_module(
#             name='elements.model', algo=algo, place=-1).create_model
#         create_loss = pkg.import_module(
#                 name='elements.loss', algo=algo, place=-1).create_loss
#         self.create_trainer = pkg.import_module(
#             name='elements.trainer', algo=algo, place=-1).create_trainer
#         self.create_actor = pkg.import_module(
#             name='elements.actor', algo=algo, place=-1).create_actor
#         self.create_buffer = pkg.import_module(
#             'elements.buffer', algo=algo).create_buffer
#         self.create_strategy = pkg.import_module(
#             'elements.strategy', algo=algo).create_strategy
#         self.create_agent = pkg.import_module(
#             'elements.agent', algo=algo).create_agent

#         self.model = create_model(config.model, env_stats, name=self._name)
#         loss = create_loss(config.loss, self.model, name=self._name)
#         self.trainer = create_trainer(config.trainer, loss, env_stats, name=self._name)
#         self.actor = create_actor(config.actor, self.model, name=self._name)

#         self.config.buffer.n_envs = env_stats.n_envs
#         self.config.buffer.state_keys = self.model.state_keys
#         self.config.buffer.use_dataset = True
#         central_buffer = config.buffer.get('central_buffer', False)
#         self.buffer = create_buffer(config.buffer, central_buffer=central_buffer)
#         am = pkg.import_module('elements.utils', algo=config.algorithm)
#         data_format = am.get_data_format(
#             self.config.trainer, env_stats, self.model)
#         self.dataset = create_dataset(self.buffer, env_stats, 
#             data_format=data_format, central_buffer=central_buffer, 
#             one_hot_action=False)

#         self.strategy = create_strategy(
#             self._name, config.strategy, actor=self.actor, 
#             trainer=self.trainer, dataset=self.dataset)
#         self.monitor = create_monitor(config.root_dir, config.model_name, self._name)

#         self.agent = create_agent(
#             config=config.agent, 
#             strategy=self.strategy, 
#             monitor=self.monitor, 
#             name=self._name, 
#             to_save_code=to_save_code
#         )

#     def name(self):
#         return self._name

#     def __getattr__(self, name):
#         if name.startswith('_'):
#             raise AttributeError(f"Attempted to get missing private attribute '{name}'")
#         if hasattr(self.strategy, name):
#             return getattr(self.strategy, name)
#         elif hasattr(self.monitor, name):
#             return getattr(self.monitor, name)
#         raise AttributeError(f"Attempted to get missing attribute '{name}'")

class RemoteAgent(RayBase):
    def build(self, builder, config):
        self.config = dict2AttrDict(config)
        elements = builder.build_agent_from_scratch(config)
        [setattr(self, k, v) for k, v in elements.items()]


def create_agent(**kwargs):
    return Agent(**kwargs)
