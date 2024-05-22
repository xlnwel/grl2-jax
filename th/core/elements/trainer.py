import os
from typing import Dict
import torch

from th.core.ckpt.base import ParamsCheckpointBase
from th.core.elements.loss import LossBase
from th.core.optimizer import build_optimizer
from th.core.typing import AttrDict, ModelPath, dict2AttrDict
from th.core.names import MODEL, OPTIMIZER
from th.core.utils import tpdv
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
    super().__init__(config, f'{name}_trainer', OPTIMIZER)
    self.aid = config.get('aid', 0)
    self.env_stats = env_stats
    self.tpdv = tpdv(config.device)

    self.loss = loss
    self.model = loss.model
    self.opts: Dict[str, torch.optim.Optimizer] = AttrDict()

    self.add_attributes()
    self.build_optimizers()
    self.post_init()

  def add_attributes(self):
    pass

  def theta_train(self):
    raise NotImplementedError

  def build_optimizers(self):
    self.opts.theta = build_optimizer(
      params=self.model.theta, 
      **self.config.theta_opt, 
      name='theta'
    )

  def train(self, data):
    raise NotImplementedError

  def post_init(self):
    """ Add some additional attributes and do some post processing here """
    pass

  """ Weights Access """
  def get_weights(self):
    weights = {
      MODEL: self.get_model_weights(),
      OPTIMIZER: self.get_optimizer_weights(),
    }
    return weights

  def set_weights(self, weights):
    self.set_model_weights(weights[MODEL])
    self.set_optimizer_weights(weights[OPTIMIZER])

  def get_model_weights(self, name: str=None):
    return self.model.get_weights(name)

  def set_model_weights(self, weights):
    self.model.set_weights(weights)

  def get_optimizer_weights(self):
    weights = dict2AttrDict({
      k: v.state_dict() for k, v in self.opts.items()
    })
    return weights

  def set_optimizer_weights(self, weights):
    for k, v in self.opts.items():
      v.load_state_dict(weights[k])

  """ Checkpoints """
  def reset_model_path(self, model_path: ModelPath):
    self.config = set_path(self.config, model_path, max_layer=0)
    self._model_path(model_path)
    self._saved_path = os.path.join(*self._model_path, self._suffix_path)

  def save_optimizer(self):
    super().save(self.opts)

  def restore_optimizer(self):
    super().restore(self.opts)

  def save(self):
    self.save_optimizer()
    self.model.save()
  
  def restore(self):
    self.restore_optimizer()
    self.model.restore()


def create_trainer(config, env_stats, loss, *, name, trainer_cls, **kwargs):
  trainer = trainer_cls(
    config=config, 
    env_stats=env_stats, 
    loss=loss, 
    name=name, 
    **kwargs)

  return trainer
