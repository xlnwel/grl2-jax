import os
import importlib
import logging
from types import FunctionType
from typing import Dict, Tuple, Set

from core.elements.actor import Actor
# from core.elements.dataset import create_dataset
from core.elements.model import Model
from core.elements.strategy import Strategy
from core.elements.trainer import TrainerBase
from core.elements.monitor import Monitor, create_monitor
from core.log import do_logging
from core.names import *
from core.typing import *
from core.utils import save_code_for_seed, save_config
from env.func import get_env_stats
from tools.timer import timeit_now
from tools.utils import set_path
from tools import pkg, yaml_op


logger = logging.getLogger(__name__)


class ElementsBuilder:
  def __init__(
    self, 
    config: dict, 
    env_stats: dict=None, 
    to_save_code: bool=False,
    name='builder', 
    max_steps=None, 
  ):
    self.config = dict2AttrDict(config, to_copy=True)
    self.env_stats = dict2AttrDict(
      get_env_stats(self.config.env) if env_stats is None else env_stats)
    self._name = name
    self._max_steps = max_steps

    self._model_path = ModelPath(self.config.root_dir, self.config.model_name)

    algo = self.config.algorithm.split('-')[-1]
    self.constructors = self.retrieve_constructor(algo)

    if to_save_code:
      timeit_now(save_code_for_seed, self.config)

  @property
  def name(self):
    return self._name
  
  def retrieve_constructor(self, algo):
    constructors = AttrDict()
    constructors.model = self._import_element(MODEL, algo).create_model
    constructors.loss = self._import_element(LOSS, algo).create_loss
    constructors.trainer = self._import_element(TRAINER, algo).create_trainer
    constructors.actor = self._import_element(ACTOR, algo).create_actor
    constructors.buffer = self._import_element(BUFFER, algo).create_buffer
    constructors.strategy = self._import_element(STRATEGY, algo).create_strategy
    constructors.agent = self._import_element(AGENT, algo).create_agent

    return constructors

  def get_model_path(self):
    return self._model_path

  """ Build Elements """
  def build_model(
    self, 
    config: dict=None, 
    env_stats: dict=None, 
    constructors: Dict[str, FunctionType]=None
  ):
    constructors = constructors or self.constructors
    config = dict2AttrDict(config or self.config)
    env_stats = dict2AttrDict(env_stats or self.env_stats)
    model = constructors.model(
      config.model, env_stats, name=config.name)
    return model

  def build_actor(
    self, 
    model: Model, 
    config: dict=None, 
    constructors: Dict[str, FunctionType]=None
  ):
    constructors = constructors or self.constructors
    config = dict2AttrDict(config or self.config)
    actor = constructors.actor(config.actor, model, name=config.name)
    
    return actor
  
  def build_trainer(
    self, 
    model: Model, 
    config: dict=None, 
    env_stats: dict=None, 
    constructors: Dict[str, FunctionType]=None
  ):
    constructors = constructors or self.constructors
    config = dict2AttrDict(config or self.config)
    env_stats = dict2AttrDict(env_stats or self.env_stats)
    loss = constructors.loss(config.loss, model, name=config.name)
    trainer = constructors.trainer(
      config.trainer, env_stats, loss, name=config.name)

    return trainer
  
  def build_buffer(
    self, 
    model: Model, 
    config: dict=None, 
    env_stats: dict=None, 
    constructors: Dict[str, FunctionType]=None, 
    **kwargs
  ):
    constructors = constructors or self.constructors
    config = dict2AttrDict(config or self.config.buffer)
    env_stats = dict2AttrDict(env_stats or self.env_stats)
    buffer = constructors.buffer(
      config, 
      model, 
      env_stats, 
      **kwargs
    )
    
    return buffer

  # def build_dataset(
  #   self, 
  #   buffer, 
  #   model: Model, 
  #   config: dict=None, 
  #   env_stats: dict=None, 
  #   central_buffer: bool=False
  # ):
  #   config = dict2AttrDict(config or self.config)
  #   env_stats = dict2AttrDict(env_stats or self.env_stats)
  #   if self.config.buffer.use_dataset:
  #     am = pkg.import_module(
  #       'elements.utils', algo=config.algorithm, place=-1)
  #     data_format = am.get_data_format(
  #       self.config.trainer, env_stats, model)
  #     buffer = create_dataset(buffer, env_stats, 
  #       data_format=data_format, central_buffer=central_buffer, 
  #       one_hot_action=False)
  #   else:
  #     buffer = buffer

  #   return buffer
  
  def build_strategy(
    self, 
    actor: Actor=None, 
    trainer: TrainerBase=None, 
    buffer=None, 
    config: dict=None, 
    env_stats: dict=None, 
    constructors: Dict[str, FunctionType]=None
  ):
    constructors = constructors or self.constructors
    config = dict2AttrDict(config or self.config)
    env_stats = dict2AttrDict(env_stats or self.env_stats)
    strategy = constructors.strategy(
      config.name, 
      config.strategy, 
      env_stats=env_stats,
      actor=actor, 
      trainer=trainer, 
      buffer=buffer, 
    )
    
    return strategy

  def build_rule_based_strategy(
    self, 
    env, 
    config, 
  ):
    path = config['path'].replace(PATH_SPLIT, '.')
    m = importlib.import_module(path)

    strategy = m.create_strategy(env, config)

    return strategy

  def build_monitor(
    self, 
    config: dict=None, 
    save_to_disk: bool=True
  ):
    if save_to_disk:
      config = dict2AttrDict(config or self.config)
      config.setdefault('monitor', {})
      return create_monitor(
        **config.monitor, 
        max_steps=self._max_steps
      )
    else:
      return create_monitor(
        use_tensorboard=False, 
        max_steps=self._max_steps
      )
  
  def build_agent(
    self, 
    strategy: Strategy, 
    monitor: Monitor=None, 
    config: dict=None, 
    constructors: FunctionType=None,
    to_restore: bool=True, 
  ):
    constructors = constructors or self.get_constructors(config)
    config = dict2AttrDict(config or self.config)
    agent = constructors.agent(
      config=config.agent, 
      strategy={config.algorithm: strategy}, 
      monitor=monitor, 
      name=config.name, 
      to_restore=to_restore, 
      builder=self
    )

    return agent

  """ Build an Strategy/Agent from Scratch .
  We delibrately define different interfaces for each type of 
  Strategy/Agent to offer default setups for a variety of situations
  
  An acting strategy/agent is used to interact with environments only
  A training strategy/agent is used for training only
  A plain strategy/agent is used for both cases
  """
  def build_acting_strategy_from_scratch(
    self, 
    config: dict=None, 
    env_stats: dict=None,
    build_monitor: bool=True, 
    to_build_for_eval: bool=False
  ):
    constructors = self.get_constructors(config)
    config = self.config if config is None else config
    env_stats = self.env_stats if env_stats is None else env_stats
    
    elements = AttrDict()
    elements.model = self.build_model(
      config=config, 
      env_stats=env_stats, 
      constructors=constructors)
    elements.actor = self.build_actor(
      model=elements.model, 
      config=config,
      constructors=constructors)
    elements.strategy = self.build_strategy(
      actor=elements.actor, 
      config=config,
      env_stats=env_stats,
      constructors=constructors)
    if build_monitor:
      elements.monitor = self.build_monitor(
        config=config, 
        save_to_disk=False)
    else:
      elements.monitor = None

    return elements
  
  def build_training_strategy_from_scratch(
    self, 
    config: dict=None, 
    env_stats: dict=None,
    build_monitor: bool=True, 
    save_monitor_stats_to_disk: bool=False,
    save_config: bool=True
  ):
    constructors = self.get_constructors(config)
    config = self.config if config is None else config
    env_stats = self.env_stats if env_stats is None else env_stats
    
    elements = AttrDict()
    elements.model = self.build_model(
      config=config, 
      env_stats=env_stats, 
      constructors=constructors)
    elements.trainer = self.build_trainer(
      model=elements.model, 
      config=config, 
      env_stats=env_stats,
      constructors=constructors)
    elements.buffer = self.build_buffer(
      model=elements.model, 
      config=config.buffer, 
      env_stats=env_stats, 
      constructors=constructors)
    # elements.buffer = self.build_dataset(
    #   buffer=elements.buffer, 
    #   model=elements.model, 
    #   config=config,
    #   env_stats=env_stats)
    elements.strategy = self.build_strategy(
      trainer=elements.trainer, 
      buffer=elements.buffer, 
      config=config,
      env_stats=env_stats,
      constructors=constructors)
    if build_monitor:
      elements.monitor = self.build_monitor(
        config=config, 
        save_to_disk=save_monitor_stats_to_disk
      )
    else:
      elements.monitor = None

    if save_config:
      self.save_config()

    return elements
  
  def build_strategy_from_scratch(
    self, 
    config: dict=None, 
    env_stats: dict=None,
    build_monitor: bool=True, 
    save_monitor_stats_to_disk: bool=False,
    save_config: bool=True
  ):
    constructors = self.get_constructors(config)
    config = self.config if config is None else config
    env_stats = self.env_stats if env_stats is None else env_stats
    
    elements = AttrDict()
    elements.model = self.build_model(
      config=config, 
      env_stats=env_stats,
      constructors=constructors)
    elements.actor = self.build_actor(
      model=elements.model, 
      config=config,
      constructors=constructors)
    elements.trainer = self.build_trainer(
      model=elements.model, 
      config=config, 
      env_stats=env_stats,
      constructors=constructors)
    elements.buffer = self.build_buffer(
      model=elements.model, 
      config=config.buffer, 
      env_stats=env_stats, 
      constructors=constructors)
    # elements.buffer = self.build_dataset(
    #   buffer=elements.buffer, 
    #   model=elements.model, 
    #   config=config,
    #   env_stats=env_stats)
    elements.strategy = self.build_strategy(
      actor=elements.actor, 
      trainer=elements.trainer, 
      buffer=elements.buffer,
      config=config,
      env_stats=env_stats,
      constructors=constructors)
    if build_monitor:
      elements.monitor = self.build_monitor(
        config=config, 
        save_to_disk=save_monitor_stats_to_disk)

    if save_config:
      self.save_config(config, env_stats)

    return elements

  def build_acting_agent_from_scratch(
    self, 
    config: dict=None, 
    env_stats: dict=None,
    build_monitor: bool=True, 
    to_build_for_eval: bool=False,
    to_restore: bool=True
  ):
    elements = self.build_acting_strategy_from_scratch(
      config=config, 
      env_stats=env_stats, 
      build_monitor=build_monitor ,
      to_build_for_eval=to_build_for_eval
    )
    elements.agent = self.build_agent(
      strategy=elements.strategy, 
      monitor=elements.monitor, 
      config=config,
      to_restore=to_restore
    )

    return elements
  
  def build_training_agent_from_scratch(
    self, 
    config: dict=None, 
    env_stats: dict=None,
    save_monitor_stats_to_disk: bool=True,
    save_config: bool=True
  ):
    elements = self.build_training_strategy_from_scratch(
      config=config, 
      env_stats=env_stats, 
      save_monitor_stats_to_disk=save_monitor_stats_to_disk,
      save_config=save_config
    )
    elements.agent = self.build_agent(
      strategy=elements.strategy, 
      monitor=elements.monitor,
      config=config,
    )

    return elements

  def build_agent_from_scratch(
    self, 
    config: dict=None, 
    env_stats: dict=None,
    save_monitor_stats_to_disk: bool=True,
    save_config: bool=True, 
    to_restore: bool=True, 
  ):
    elements = self.build_strategy_from_scratch(
      config=config, 
      env_stats=env_stats, 
      save_monitor_stats_to_disk=save_monitor_stats_to_disk, 
      save_config=save_config, 
    )
    elements.agent = self.build_agent(
      strategy=elements.strategy, 
      monitor=elements.monitor, 
      config=config, 
      to_restore=to_restore
    )
    
    return elements

  """ Configuration Operations """
  def get_config(self):
    return self.config

  def get_constructors(self, config: AttrDict):
    if config is not None and config.algorithm != self.config.algorithm:
      constructors = self.retrieve_constructor(config.algorithm)
    else:
      constructors = self.constructors
    return constructors

  def save_config(self, config: dict=None, env_stats: dict=None):
    config = config or self.config
    env_stats = env_stats or self.env_stats
    config.env_stats = env_stats.copy()
    config.env_stats.pop('action_space')
    config.env_stats.pop('obs_dtype')
    config.env_stats.pop('action_dtype')
    save_config(config)
    model = ModelPath(self.config.root_dir, self.config.model_name)
    do_logging(f'Save config: {model}', backtrack=3, color='green')

  """ Implementations"""
  def _import_element(self, name, algo=None):
    try:
      module = pkg.import_module(f'elements.{name}', algo=algo)
    except Exception as e:
      level = 'info' if name == 'agent' else 'pwc'
      do_logging(
        f'Switch to default module({name}) due to error: {e}', 
        logger=logger, level=level, backtrack=4)
      do_logging(
        "You are safe to neglect it if it's an intended behavior. ", 
        logger=logger, level=level, backtrack=4)
      module = pkg.import_module(f'elements.{name}', pkg='core')
    return module


""" Element Builder with Version Control """
class ElementsBuilderVC(ElementsBuilder):
  def __init__(
    self, 
    config: dict, 
    env_stats: dict=None, 
    start_iteration=0, 
    to_save_code=False, 
    name='builder', 
  ):
    super().__init__(
      config, 
      env_stats=env_stats, 
      to_save_code=to_save_code, 
      name=name
    )

    self._default_model_path = self._model_path
    self._builder_path = os.path.join(*self._model_path, f'{self._name}.yaml')

    self._iteration = start_iteration
    self._all_versions: Set = set()
    self.restore()

  """ Version Control """  
  def get_iteration(self):
    return self._iteration

  def set_iteration(self, iteration):
    self._iteration = iteration
    version = iteration
    self._all_versions.add(version)
    self.config.iteration = iteration
    self.config.version = version

    root_dir = self.config.root_dir
    model_name = construct_model_name_from_version(
      self._default_model_path.model_name, 
      self._iteration, 
      version
    )
    self._model_path = ModelPath(root_dir, model_name)
    self.config = set_path(self.config, self._model_path)
    self._all_versions.add(version)
    self.save_config()
    self.save()

  def get_sub_version(self, config: AttrDict, iteration: int) -> Tuple[ModelPath, AttrDict]:
    self._iteration = iteration
    def compute_next_version(version: str):
      if '.' in version:
        base_version, sub_version = version.rsplit('.', maxsplit=1)
        sub_version = eval(sub_version)
        version = '.'.join([base_version, str(sub_version+1)])
      else:
        base_version = version
        sub_version = 1
        version = '.'.join([base_version, str(sub_version)])

      if version in self._all_versions:
        base_version = '.'.join([base_version, str(sub_version)])
        sub_version = 1
        version = '.'.join([base_version, str(sub_version)])
        while version in self._all_versions:
          sub_version += 1
          version = '.'.join([base_version, str(sub_version)])

      return version

    root_dir = config.root_dir
    version = get_vid(config.model_name)
    assert version == f'{config.version}', (version, config.version)
    version = compute_next_version(version)
    config.iteration = self._iteration
    config.version = version

    model_name = construct_model_name_from_version(
      self._default_model_path.model_name, 
      self._iteration, 
      version
    )
    model_path = ModelPath(root_dir, model_name)
    model_dir = os.path.join(*model_path)
    config = set_path(config, model_path)
    if not os.path.isdir(model_dir):
      os.makedirs(model_dir, exist_ok=True)
    self._all_versions.add(version)
    self.save_config(config)
    self.save()

    return model_path, config

  """ Save & Restore """
  def restore(self):
    if os.path.exists(self._builder_path):
      data = yaml_op.load(self._builder_path)
      self._iteration = data['iteration']
      self._all_versions = data['all_versions']
      self.config = dict2AttrDict(data['config'])

  def save(self):
    yaml_op.dump(
      self._builder_path, 
      iteration=self._iteration, 
      all_versions=self._all_versions, 
      config=AttrDict2dict(self.config), 
    )

if __name__ == '__main__':
  from env.func import create_env
  config = yaml_op.load_config('algo/zero/configs/card.yaml')
  env = create_env(config['env'])
  config.model_name = 'test'
  env_stats = env.stats()
  builder = ElementsBuilder(config, env_stats=env_stats, name='zero', incremental_version=True)
  elements = builder.build_agent_from_scratch()
  elements.agent.save()
  builder.increase_version()
  elements = builder.build_agent_from_scratch()
  elements.agent.save()
