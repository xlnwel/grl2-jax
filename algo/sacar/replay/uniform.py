import numpy as np

from utility.decorators import override
from algo.sacar.replay.base import Replay
from algo.sacar.replay.utils import init_buffer


class UniformReplay(Replay):
    """ Implementation """
    @override(Replay)
    def _sample(self):
        size = self.capacity if self.is_full else self.mem_idx
        indexes = np.random.randint(0, size, self.batch_size)
        
        samples = self._get_samples(indexes)

        return samples
