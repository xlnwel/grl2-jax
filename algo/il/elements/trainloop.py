import collections
import numpy as np

from core.typing import AttrDict
from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
  def _train(self):
    n = 0
    stats = AttrDict()

    if self.config.ergodic:
      for data in self.buffer.ergodic_sample(n=self.config.training_data_size):
        if data is None:
          break
        stats = self._train_with_data(data)
        n += 1
      self.training_data = data
    else:
      data = self.sample_data()
      if data is None:
        return 0, AttrDict()
      stats = self._train_with_data(data)
      n += 1

    return n, stats
