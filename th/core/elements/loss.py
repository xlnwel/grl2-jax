from th.core.elements.model import Model
from th.core.typing import AttrDict


class LossBase:
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
    self.post_init()

  def loss(self):
    raise NotImplementedError

  def post_init(self):
    """ Add some additional attributes and do some post processing here """
    pass
