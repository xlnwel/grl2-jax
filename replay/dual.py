import numpy as np

from utility.utils import to_int
from replay.base import Replay
from replay.uniform import UniformReplay
from replay.per import ProportionalPER


class DualReplay(Replay):
    def __init__(self, config, *keys, state_shape=None):
        self._type = config['type']
        self.capacity = to_int(config['capacity'])
        self.min_size = to_int(config['min_size'])
        self.batch_size = config['batch_size']

        BufferType = ProportionalPER if self._type.endswith('proportional') else UniformReplay
        good_frac = config['good_frac']
        config['capacity'] = int(self.capacity * good_frac)
        config['min_size'] = int(self.min_size * good_frac)
        config['batch_size'] = int(self.batch_size * good_frac)
        self.good_replay = BufferType(config, *keys, state_shape=state_shape)

        config['capacity'] = self.capacity - config['capacity']
        config['min_size'] = self.min_size - config['min_size']
        config['batch_size'] = self.batch_size - config['batch_size']
        self.regular_replay = BufferType(config, *keys, state_shape=state_shape)

    def buffer_type(self):
        return self._type
        
    def good_to_learn(self):
        return self.good_replay.good_to_learn() and self.regular_replay.good_to_learn()

    def __len__(self):
        return self.capacity if self.is_full else len(self.good_replay) + len(self.regular_replay)

    def sample(self):
        assert self.good_to_learn()

        good_samples = self.good_replay.sample()
        regular_samples = self.regular_replay.sample()

        return self.combine_samples(good_samples, regular_samples)

    def combine_samples(self, samples1, samples2):
        samples = []
        for i in range(len(samples1)):
            if isinstance(samples1[i], np.ndarray):
                assert type(samples2[i]) is type(samples1[i])
                samples.append(np.concatenate([samples1[i], samples2[i]]))
                assert samples[-1].shape[0] == self.batch_size
                assert samples[-1].shape[1:] == samples1[i].shape[1:] == samples2[i].shape[1:], f'{i} {samples[-1].shape}, {samples1[i].shape}, {samples2[i].shape}'
                assert samples[-1].dtype == samples1[i].dtype == samples2[i].dtype, f'{i} {samples[-1].dtype}, {samples1[i].dtype}, {samples2[i].dtype}'
            else:
                assert isinstance(samples1[i], [list, tuple])
                assert isinstance(samples2[i], [list, tuple])
                samples.append(self.combine_samples(samples1[i], samples2[i]))

        return tuple(samples)

    def merge(self, local_buffer, length, dest_replay):
        if dest_replay == 'good_replay':
            self.good_replay.merge(local_buffer, length)
        elif dest_replay == 'regular_replay':
            self.regular_replay.merge(local_buffer, length)
        else:
            raise NotImplementedError
