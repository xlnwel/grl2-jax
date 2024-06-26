from typing import Tuple, Union
import jax

from core.elements.actor import Actor
from core.elements.model import Model
from core.elements.trainer import TrainerBase, TrainerEnsemble
from core.elements.trainloop import TrainingLoop
from core.mixin.strategy import StepCounter, Memory
from core.names import *
from core.typing import ModelPath, AttrDict, dict2AttrDict
from env.typing import EnvOutput
from tools.run import concat_along_unit_dim
from tools.utils import batch_dicts, set_path
from tools import pkg


class Strategy:
  """ Initialization """
  def __init__(
    self, 
    *, 
    name: str,
    config: AttrDict,
    env_stats: AttrDict,
    trainer: Union[TrainerBase, TrainerEnsemble]=None, 
    actor: Actor=None,
    train_loop: TrainingLoop=None,
  ):
    self._name = name
    self.config = config
    self.env_stats = env_stats
    if trainer is None and actor is None:
      raise RuntimeError('Neither trainer nor actor is provided')

    self.model: Model = actor.model if trainer is None else trainer.model
    self.trainer: TrainerBase = trainer
    self.actor: Actor = actor
    self.train_loop: TrainingLoop = train_loop

    self._setup_memory_cls()
    self._memory = self.memory_cls(self.model)

    if self.config.get('root_dir'):
      self._model_path = ModelPath(self.config.root_dir, self.config.model_name)
    else:
      self._model_path = None
    self.step_counter = StepCounter(self._model_path, name=f'{self._name}_step_counter')

    self._post_init()

  def _setup_memory_cls(self):
    self.memory_cls = Memory
    
  def _post_init(self):
    pass

  @property
  def is_trainable(self):
    return self.trainer is not None

  @property
  def name(self):
    return self._name

  def reset_model_path(self, model_path: ModelPath):
    """ 设置模型路径

    Args:
        model_path (ModelPath): (模型根目录, 模型子目录)
    """
    self._model_path = model_path
    self.step_counter = StepCounter(
      model_path, 
      name=f'{self._name}_step_counter'
    )
    self.config = set_path(self.config, model_path, max_layer=0)
    if self.model is not None:
      self.model.reset_model_path(model_path)
    if self.actor is not None:
      self.actor.reset_model_path(model_path)
    if self.trainer is not None:
      self.trainer.reset_model_path(model_path)

  def get_model_path(self):
    """ 获取模型路径 """
    return self._model_path

  def __getattr__(self, name):
    """ 读取step_counter(训练/环境步数计数器)和train_loop(训练流程)的接口和属性.
    我们不暴露其他模块的接口, 推荐直接通过模块实例调用

    Args:
        name (str): 接口/属性名
    """
    # Do not expose the interface of independent elements here. 
    # Invoke them directly instead
    if name.startswith('_'):
      raise AttributeError(f"Attempted to get missing private attribute '{name}'")
    elif hasattr(self.step_counter, name):
      return getattr(self.step_counter, name)
    elif self.train_loop is not None and hasattr(self.train_loop, name):
      return getattr(self.train_loop, name)
    raise AttributeError(f"Attempted to get missing attribute '{name}'")

  def get_weights(self, module_name=None, opt_weights=True, aux_stats=True,
      train_step=False, env_step=False):
    """ 获取权重

    Args:
        module_name (_type_, optional): 模块名. Defaults to None.
        opt_weights (bool, optional): 是否返回优化器权重. Defaults to True.
        aux_stats (bool, optional): 是否返回辅助参数. Defaults to True.
        train_step (bool, optional): 是否返回训练步数. Defaults to False.
        env_step (bool, optional): 是否返回环境交互步数. Defaults to False.

    Returns:
        Dict: 包含权重的字典
    """
    weights = {}
    if self.model is not None:
      weights[MODEL] = jax.device_get(self.model.get_weights(module_name))
    if self.trainer is not None and opt_weights:
      weights[OPTIMIZER] = jax.device_get(self.trainer.get_optimizer_weights())
    if self.actor is not None and aux_stats:
      weights[ANCILLARY] = self.actor.get_auxiliary_stats()
    if train_step:
      weights[f'train_step'] = self.step_counter.get_train_step()
    if env_step:
      weights[f'env_step'] = self.step_counter.get_env_step()

    return weights

  def set_weights(self, weights):
    """ 设置权重

    Args:
        weights (Dict): 包含权重的字典
    """
    if MODEL in weights:
      self.model.set_weights(weights[MODEL])
    if OPTIMIZER in weights and self.trainer is not None:
      self.trainer.set_optimizer_weights(weights[OPTIMIZER])
    if ANCILLARY in weights:
      self.actor.set_auxiliary_stats(weights[ANCILLARY])
    if 'train_step' in weights:
      self.step_counter.set_train_step(weights['train_step'])
    if 'env_step' in weights:
      self.step_counter.set_env_step(weights['env_step'])

  def train(self, **kwargs):
    """ 训练模型

    Args:
        **kwargs: 训练参数

    Returns:
        Dict: 包含训练统计信息的字典
    """
    n, stats = self.train_loop.train(
      self.step_counter.get_train_step(), **kwargs)
    self.step_counter.add_train_step(n)
    return stats

  """ Memory Management """
  def build_memory(self, for_self=False):
    """ 构建RNN记忆模块

    Args:
        for_self (bool, optional): 是否为自己构建. Defaults to False.

    Returns:
        _type_: RNN的记忆模块
    """
    memory = self.memory_cls(self.model)
    if for_self:
      self._memory = memory
    return memory

  def get_states(self):
    """ 获取记忆模块的状态

    Returns:
        NamedTuple: 记忆模块的状态
    """
    return self._memory.get_states()

  def reset_states(self):
    """ 重置记忆模块的状态

    Returns:
        NamedTuple: 记忆模块的状态
    """
    state = self._memory.get_states()
    self._memory.reset_states()
    return state

  def set_states(self, state=None):
    """ 设置记忆模块的状态

    Args:
        state (NamedTuple, optional): 记忆模块的状态. Defaults to None.
    """
    self._memory.set_states(state=state)
  
  def get_memory(self):
    """ 获取记忆模块 """
    return self._memory
  
  def reset_memory(self):
    """ 重置记忆模块, 并返回之前的记忆模块 """
    memory = self._memory
    self._memory = self.memory_cls(self.model)
    return memory
  
  def set_memory(self, memory: Memory):
    """ 设置记忆模块, 并返回之前的记忆模块 """
    old_memory = self._memory
    self._memory = memory
    return old_memory

  """ Call """
  def __call__(self, env_output: EnvOutput, evaluation: bool=False):
    """根据环境返回信息作推理

    Args:
        env_output (EnvOutput): 环境返回信息
        evaluation (bool, optional): 是否评估模式. Defaults to False.

    Returns:
        Dict: 动作
        Dict: 其他推理产生的统计数据
    """
    inp = self._prepare_input_to_actor(env_output)
    out = self.actor(inp, evaluation=evaluation)
    out = self._record_output(out)
    return out[:2]

  def _prepare_input_to_actor(self, env_output: EnvOutput):
    """ 提取EnvOutput中的信息, 返回数据字典用于Actor推理

    Args:
        env_output (EnvOutput): 环境输出

    Returns:
        Dict: 数据字典
    """
    if isinstance(env_output.obs, list):
      inp = batch_dicts(env_output.obs, concat_along_unit_dim)
      reset = concat_along_unit_dim(env_output.reset)
    else:
      inp = env_output.obs
      reset = env_output.reset
    inp = self._memory.add_memory_state_to_input(inp, reset)
    inp = dict2AttrDict(inp, to_copy=True)
    return inp

  def _record_output(self, out: Tuple):
    """ 记录训练输出内容 """
    state = out[-1]
    self._memory.set_states(state)
    return out

  """ Checkpoint Ops """
  def restore(self, skip_model=False, skip_actor=False, skip_trainer=False):
    """ 恢复策略

    Args:
        skip_model (bool, optional): 是否跳过模型. Defaults to False.
        skip_actor (bool, optional): 是否跳过Actor的参数. Defaults to False.
        skip_trainer (bool, optional): 是否跳过Trainer得参数. Defaults to False.
    """
    if not skip_model and self.model is not None:
      self.model.restore()
    if not skip_actor and self.actor is not None:
      self.actor.restore_auxiliary_stats()
    if not skip_trainer and self.trainer is not None:
      self.trainer.restore_optimizer()
    self.step_counter.restore_step()

  def save(self):
    """ 保存策略 """
    if self.model is not None:
      self.model.save()
    if self.actor is not None:
      self.actor.save_auxiliary_stats()
    if self.trainer is not None:
      self.trainer.save_optimizer()
    self.step_counter.save_step()


def create_strategy(
  name, 
  config: AttrDict,
  env_stats: AttrDict, 
  actor: Actor=None,
  trainer: Union[TrainerBase, TrainerEnsemble]=None, 
  buffer=None,
  *,
  strategy_cls=Strategy,
  training_loop_cls=None
):
  if trainer is not None:
    if buffer is None:
      raise ValueError('Missing buffer')
    if training_loop_cls is None:
      algo = config.algorithm.split('-')[-1]
      training_loop_cls = pkg.import_module(
        'elements.trainloop', algo=algo).TrainingLoop
    train_loop = training_loop_cls(
      config=config.train_loop, 
      buffer=buffer, 
      trainer=trainer
    )
  else:
    train_loop = None

  strategy = strategy_cls(
    name=name,
    config=config,
    env_stats=env_stats,
    trainer=trainer,
    actor=actor,
    train_loop=train_loop
  )

  return strategy
