import jax

from core.elements.trainer import Trainer as TrainerBase
from jx.elements.loss import Loss
from jx.elements.optimizer import build_optimizer
from core.typing import AttrDict
from tools.timer import Timer


class Trainer(TrainerBase):
  def __init__(
    self, 
    *,
    config: AttrDict,
    env_stats: AttrDict,
    loss: Loss,
    name: str
  ):
    super().__init__(config=config, env_stats=env_stats, loss=loss, name=name)

  def dl_init(self):
    self.rng = self.model.rng

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
  
  def haiku_tabulate(self):
    pass

  """ Weights Access """
  def get_optimizer_weights(self):
    weights = self.params.asdict(shallow=True)
    return weights

  def set_optimizer_weights(self, weights):
    assert set(weights).issubset(set(self.params)) or set(self.params).issubset(set(weights)), (list(self.params), list(weights))
    for k, v in weights.items():
      assert len(self.params[k]) == len(v), (k, len(self.params[k], len(v)))
      self.params[k] = v


def create_trainer(config, env_stats, loss, *, name, trainer_cls, **kwargs):
  trainer = trainer_cls(
    config=config, 
    env_stats=env_stats, 
    loss=loss, 
    name=name, 
    **kwargs)

  return trainer
