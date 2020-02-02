import numpy as np

from utility.decorators import override
from replay.base import Replay
from replay.utils import init_buffer


class UniformReplay(Replay):
    """ Implementation """
    @override(Replay)
    def _sample(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        size = self.capacity if self.is_full else self.mem_idx
        indexes = np.random.randint(0, size, batch_size)
        
        samples = self._get_samples(indexes)

        return samples
