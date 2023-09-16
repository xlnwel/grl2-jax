from typing import Dict, Sequence, Union
import jax
import optax

from core.ckpt.base import ParamsCheckpointBase
from core.elements.loss import LossBase
from core.ensemble import Ensemble
from core.optimizer import build_optimizer
from core.typing import ModelPath, dict2AttrDict
from core.typing import AttrDict
from tools.timer import Timer
from tools.utils import set_path


class TrainerBase(ParamsCheckpointBase):
  def __init__(
    self, 
    *,
    config: AttrDict,
    env_stats: AttrDict,
    loss: LossBase,
    name: str
  ):
    super().__init__(config, name=f'{name}_trainer')
    self.env_stats = env_stats

    self.loss = loss
    self.model = loss.model
    self.opts: Dict[str, optax.GradientTransformation] = AttrDict()
    self.opt_names: Dict[str, str] = AttrDict()
    self.rng = self.model.rng

    self.add_attributes()
    self.build_optimizers()
    self.compile_train()
    self.post_init()

  def add_attributes(self):
    pass

  def theta_train(self):
    raise NotImplementedError

  def build_optimizers(self):
    self.opts.theta, self.params.theta = build_optimizer(
      params=self.model.theta, 
      **self.config.theta_opt, 
      name='theta'
    )

  def compile_train(self):
    with Timer(f'{self.name}_jit_train', 1):
      _jit_train = jax.jit(self.theta_train, static_argnames='return_stats')
    def jit_train(*args, return_stats=True, **kwargs):
      self.rng, rng = jax.random.split(self.rng)
      return _jit_train(*args, rng=rng, return_stats=return_stats, **kwargs)
    self.jit_train = jit_train
    self.haiku_tabulate()

  def haiku_tabulate(self, data=None):
    pass

  def train(self, data):
    raise NotImplementedError

  def post_init(self):
    """ Add some additional attributes and do some post processing here """
    pass

  """ Weights Access """
  def get_weights(self):
    weights = {
      'model': self.model.get_weights(),
      'opt': self.get_optimizer_weights(),
    }
    return weights

  def set_weights(self, weights):
    self.model.set_weights(weights['model'])
    self.set_optimizer_weights(weights['opt'])

  def get_model_weights(self, name: str=None):
    return self.model.get_weights(name)

  def set_model_weights(self, weights):
    self.model.set_weights(weights)

  def get_optimizer_weights(self):
    weights = self.params.asdict(shallow=True)
    return weights

  def set_optimizer_weights(self, weights):
    assert set(weights).issubset(set(self.params)) or set(self.params).issubset(set(weights)), (list(self.params), list(weights))
    for k, v in weights.items():
      assert len(self.params[k]) == len(v), (k, len(self.params[k], len(v)))
      self.params[k] = v

  """ Checkpoints """
  def reset_model_path(self, model_path: ModelPath):
    self.config = set_path(self.config, model_path, max_layer=0)
    self._ckpt.reset_model_path(model_path)

  def save_optimizer(self):
    self._ckpt.save(self.params)

  def restore_optimizer(self):
    self._ckpt.restore()

  def save(self):
    self.save_optimizer()
    self.model.save()
  
  def restore(self):
    self.restore_optimizer()
    self.model.restore()


class TrainerEnsemble(Ensemble):
  def __init__(
    self, 
    *, 
    config: AttrDict, 
    env_stats: AttrDict,
    components: Dict[str, TrainerBase], 
    name: str, 
  ):
    super().__init__(
      config=config, 
      env_stats=env_stats, 
      components=components, 
      name=f'{name}_trainer', 
    )

    self.model = dict2AttrDict({
      k: v.model for k, v in components.items()
    }, shallow=True)

  """ Weights Access """
  def get_weights(self, names: Union[str, Sequence]=None):
    names = self._get_names(names)
    weights = {
      k: v.get_weights() for k, v in self.components.items()
    }
    return weights

  def set_weights(self, weights):
    assert set(weights).issubset(set(self.components)) or set(self.components).issubset(set(weights)), (list(self.components), list(weights))
    for k, v in weights.items():
      self.components[k].set_weights(v)

  def get_model_weights(self, names: Union[str, Sequence]=None):
    names = self._get_names(names)
    weights = {
      k: self.components[k].get_model_weights() for k in names
    }
    return weights

  def set_model_weights(self, weights):
    assert set(weights).issubset(set(self.components)) or set(self.components).issubset(set(weights)), (list(self.components), list(weights))
    for k, v in weights.items():
      self.components[k].set_model_weights(v)

  def get_optimizer_weights(self, names: Union[str, Sequence]=None):
    names = self._get_names(names)
    weights = {
      k: self.components[k].get_optimizer_weights() for k in names
    }
    return weights

  def set_optimizer_weights(self, weights):
    assert set(weights).issubset(set(self.components)) or set(self.components).issubset(set(weights)), (list(self.components), list(weights))
    for k, v in weights.items():
      self.components[k].set_optimizer_weights(v)

  """ Checkpoints """
  def save_optimizer(self):
    for v in self.components.values():
      v.save_optimizer()

  def restore_optimizer(self):
    for v in self.components.values():
      v.restore_optimizer()


def create_trainer(config, env_stats, loss, *, name, trainer_cls, **kwargs):
  trainer = trainer_cls(
    config=config, 
    env_stats=env_stats, 
    loss=loss, 
    name=name, 
    **kwargs)

  return trainer
