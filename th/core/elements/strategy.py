from typing import Tuple, Union

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
      self._model_path = ModelPath(
        self.config.root_dir, 
        self.config.model_name
      )
    else:
      self._model_path = None
    self.step_counter = StepCounter(
      self._model_path, 
      name=f'{self._name}_step_counter'
    )

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
    return self._model_path

  def __getattr__(self, name):
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
    weights = {}
    if self.model is not None:
      weights[MODEL] = self.model.get_weights(module_name)
    if self.trainer is not None and opt_weights:
      weights[OPTIMIZER] = self.trainer.get_optimizer_weights()
    if self.actor is not None and aux_stats:
      weights[ANCILLARY] = self.actor.get_auxiliary_stats()
    if train_step:
      weights[f'train_step'] = self.step_counter.get_train_step()
    if env_step:
      weights[f'env_step'] = self.step_counter.get_env_step()

    return weights

  def set_weights(self, weights):
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

  def train_record(self, **kwargs):
    n, stats = self.train_loop.train(
      self.step_counter.get_train_step(), **kwargs)
    self.step_counter.add_train_step(n)
    return stats

  """ Memory Management """
  def build_memory(self, for_self=False):
    memory = self.memory_cls(self.model)
    if for_self:
      self._memory = memory
    return memory

  def get_states(self):
    return self._memory.get_states()

  def reset_states(self):
    state = self._memory.get_states()
    self._memory.reset_states()
    return state

  def set_states(self, state=None):
    self._memory.set_states(state=state)
  
  def get_memory(self):
    return self._memory
  
  def reset_memory(self):
    memory = self._memory
    self._memory = self.memory_cls(self.model)
    return memory
  
  def set_memory(self, memory: Memory):
    old_memory = self._memory
    self._memory = memory
    return old_memory

  """ Call """
  def __call__(
    self, 
    env_output: EnvOutput, 
    evaluation: bool=False,
  ):
    inp = self._prepare_input_to_actor(env_output)
    out = self.actor(inp, evaluation=evaluation)
    out = self._record_output(out)
    return out[:2]

  def _prepare_input_to_actor(self, env_output: EnvOutput):
    """ Extract data from env_output as the input 
    to Actor for inference """
    if isinstance(env_output.obs, list):
      inp = batch_dicts(env_output.obs, concat_along_unit_dim)
    else:
      inp = env_output.obs
    inp = dict2AttrDict(inp, to_copy=True)
    return inp

  def _record_output(self, out: Tuple):
    """ Record some data in out """
    return out

  """ Checkpoint Ops """
  def restore(self, skip_model=False, skip_actor=False, skip_trainer=False):
    if not skip_model and self.model is not None:
      self.model.restore()
    if not skip_actor and self.actor is not None:
      self.actor.restore_auxiliary_stats()
    if not skip_trainer and self.trainer is not None:
      self.trainer.restore_optimizer()
    self.step_counter.restore_step()

  def save(self):
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