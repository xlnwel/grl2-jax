from core.elements.trainer import Trainer as TrainerBase
from th.elements.optimizer import build_optimizer
from th.utils import tpdv


class Trainer(TrainerBase):
  def dl_init(self):
    self.tpdv = tpdv(self.config.device)

  def build_optimizers(self):
    self.opts.theta = build_optimizer(
      params=self.model.theta, 
      **self.config.theta_opt, 
      name='theta'
    )

  """ Weights Access """
  def get_optimizer_weights(self):
    weights = {
      k: v.state_dict() for k, v in self.opts.items()
    }
    return weights

  def set_optimizer_weights(self, weights):
    for k, v in self.opts.items():
      v.load_state_dict(weights[k])


def create_trainer(config, env_stats, loss, *, name, trainer_cls, **kwargs):
  trainer = trainer_cls(
    config=config, 
    env_stats=env_stats, 
    loss=loss, 
    name=name, 
    **kwargs)

  return trainer
