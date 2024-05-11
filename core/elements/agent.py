import os
from typing import Dict, Union

from core.decorator import *
from core.elements.builder import ElementsBuilder
from core.elements.strategy import Strategy
from tools.log import do_logging
from core.names import PATH_SPLIT
from core.elements.monitor import Monitor
from core.typing import ModelPath, get_algo, AttrDict, ModelWeights
from run.utils import search_for_config


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
    self.strategy: Strategy = next(iter(self.strategies.values()))
    self.monitor: Monitor = monitor
    self.builder: ElementsBuilder = builder
    # trainable is set to align with the first strategy
    self.is_trainable = self.strategy.is_trainable

    if to_restore:
      self.restore()

    self._post_init()
  
  def _post_init(self):
    pass

  @property
  def name(self):
    return self._name

  def reset_model_path(self, model_path: ModelPath):
    """ 设置模型路径

    Args:
        model_path (ModelPath): (模型根目录, 模型子目录)
    """
    self.strategy.reset_model_path(model_path)
    if self.monitor:
      self.monitor.reset_model_path(model_path)

  def get_model_path(self):
    """ 获取模型路径 """
    return self._model_path

  def add_strategy(self, sid: str, strategy: Strategy):
    """添加策略(strategy)

    Args:
        sid (str): 策略标识
        strategy (Strategy): 策略实例
    """
    self.strategies[sid] = strategy

  def switch_strategy(self, sid: str):
    """切换策略(strategy)

    Args:
        sid (str): 策略标识
    """
    self.strategy = self.strategies[sid]

  def set_strategy(self, strategy: ModelWeights, *, env=None):
    """设置策略

    Args:
        strategy (ModelWeights): (模型, 权重)元组, 当策略是规则时, 权重指定规则策略所定义的地址
        env (Env, optional): 训练环境. 默认为None.
    """
    self._model_path = strategy.model
    if len(strategy.model.root_dir.split(PATH_SPLIT)) < 3:
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
      algo = get_algo(strategy.model.root_dir)
      if algo not in self.strategies:
        config = search_for_config(os.path.join(strategy.model))
        self.config = config
        build_func = self.builder.build_training_strategy_from_scratch \
          if self.is_trainable else self.builder.build_acting_strategy_from_scratch
        elements = build_func(
          config=config, 
          env_stats=self.strategy.env_stats, 
          build_monitor=self.monitor is not None
        )
        self.strategies[algo] = elements.strategy
        do_logging(f'Adding new strategy {strategy.model}')
      if self.monitor is not None and self.monitor.save_to_disk:
        self.monitor = self.monitor.reset_model_path(strategy.model)
      self.strategies[algo].set_weights(strategy.weights)

    self.strategy = self.strategies[algo]

  def __getattr__(self, name: str):
    """读取策略和监视器的接口/属性

    Args:
        name (str): 接口/属性名
    """
    if name.startswith('_'):
      raise AttributeError(f"Attempted to get missing private attribute '{name}'")
    if hasattr(self.strategy, name):
      # Expose the interface of strategy as Agent and Strategy are interchangeably in many cases 
      return getattr(self.strategy, name)
    elif hasattr(self.monitor, name):
      return getattr(self.monitor, name)
    raise AttributeError(f"Attempted to get missing attribute '{name}'")

  def __call__(self, *args, **kwargs):
    """ 调用策略 """
    return self.strategy(*args, **kwargs)

  """ Train """
  def train_record(self, **kwargs):
    """ 训练并记录训练产生的统计数据

    Returns:
        Dict: 统计数据
    """
    stats = self.strategy.train(**kwargs)
    self.monitor.store(**stats)
    return stats

  def save(self):
    """ 保存策略 """
    for s in self.strategies.values():
      s.save()
  
  def restore(self, skip_model=False, skip_actor=False, skip_trainer=False):
    """ 恢复策略

    Args:
        skip_model (bool, optional): 是否跳过模型. Defaults to False.
        skip_actor (bool, optional): 是否跳过Actor的参数. Defaults to False.
        skip_trainer (bool, optional): 是否跳过Trainer得参数. Defaults to False.
    """
    for s in self.strategies.values():
      s.restore(
        skip_model=skip_model, 
        skip_actor=skip_actor, 
        skip_trainer=skip_trainer
      )


def create_agent(**kwargs):
  """ 创建 Agent """
  return Agent(**kwargs)
