from core.elements.model import Model, Ensemble
from core.typing import AttrDict, dict2AttrDict


class Loss:
  def __init__(
    self,
    *,
    config: AttrDict,
    model: Model,
    name: str
  ):
    self.config = config
    self.name = name

    self.model = model
    self.modules = model.modules

    self.dl_init()
    self.post_init()

  def loss(self):
    raise NotImplementedError

  def dl_init(self):
    pass

  def post_init(self):
    """ Add some additional attributes and do some post processing here """
    pass


class LossEnsemble(Ensemble):
  def __init__(
    self, 
    *, 
    config: AttrDict, 
    components=None, 
    name, 
  ):
    super().__init__(
      config=config,
      components=components, 
      name=name,
    )
    self.model = dict2AttrDict({
      k: v.model for k, v in components.items()
    }, shallow=True)
