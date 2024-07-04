from typing import Dict, Sequence, Union

from core.ckpt.base import ParamsCheckpointBase
from core.elements.loss import Loss
from core.ensemble import Ensemble
from core.typing import AttrDict, ModelPath, dict2AttrDict
from core.names import MODEL, OPTIMIZER
from tools.utils import set_path


class Trainer(ParamsCheckpointBase):
  def __init__(
    self, 
    *,
    config: AttrDict,
    env_stats: AttrDict,
    loss: Loss,
    name: str
  ):
    super().__init__(config, f'{name}_trainer', OPTIMIZER)
    self.aid = config.get('aid', 0)
    self.env_stats = env_stats

    self.loss = loss
    self.model = loss.model
    self.opts: Dict = AttrDict()

    self.dl_init()
    self.post_init()
    self.build_optimizers()
    self.compile_train()

  def dl_init(self):
    """ Add some dl-specific attributes here """
    pass

  def post_init(self):
    """ Add some additional attributes and do some post processing here """
    pass

  def theta_train(self):
    raise NotImplementedError

  def build_optimizers(self):
    raise NotImplementedError

  def compile_train(self):
    pass

  def train(self, data):
    raise NotImplementedError

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
    params = self.get_optimizer_weights()
    self._ckpt.save(params)

  def restore_optimizer(self):
    params = self._ckpt.restore()
    self.set_optimizer_weights(params)

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
    components: Dict[str, Trainer], 
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
