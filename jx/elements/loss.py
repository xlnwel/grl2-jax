from core.elements.loss import Loss as LossBase
from core.typing import AttrDict, dict2AttrDict


class Loss(LossBase):
  def dl_init(self):
    self.params = self.model.params
