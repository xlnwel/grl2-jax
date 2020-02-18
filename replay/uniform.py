import numpy as np

from utility.decorators import override
from replay.base import Replay
from replay.utils import init_buffer


class UniformReplay(Replay):
    @override(Replay)
    def sample(self, batch_size=None):
        assert self.good_to_learn(), (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self.min_size})')

        samples = self._sample(batch_size)

        return samples

    """ Implementation """
    @override(Replay)
    def _sample(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        size = self.capacity if self.is_full else self.mem_idx
        indexes = np.random.randint(0, size, size=batch_size)
        
        samples = self._get_samples(indexes)

        return samples
