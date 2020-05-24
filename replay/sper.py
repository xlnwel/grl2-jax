import math
import copy
import collections
import numpy as np

from replay.ds.sum_tree import SumTree
from replay.per import ProportionalPER


class SequentialPER(ProportionalPER):
    def __init__(self, config, state_keys=[]):
        super().__init__(config)
        self._dtype = {16: np.float16, 32: np.float32}[self._precision]
        self._state_keys = state_keys
        self._data_structure = SumTree(self._capacity)
        self._temp_buff = collections.defaultdict(list)
        self._memory = collections.deque(maxlen=self._capacity)
        self._pop_size = self._sample_size - self._burn_in_size
        self._tb_idx = 0

    def __len__(self):
        return len(self._memory)

    def add(self, **kwargs):
        if self._temp_buff == {}:
            for k in kwargs:
                n_states = math.ceil(self._sample_size / self._pop_size)
                self._temp_buff[k] = collections.deque(
                    maxlen=n_states if k in self._state_keys else self._sample_size)
        else:
            np.testing.assert_equal(set(self._temp_buff), set(kwargs))
        for k, v in kwargs.items():
            if k in self._state_keys:
                if self._tb_idx % self._pop_size == 0:
                    self._temp_buff[k].append(v)
            else:
                self._temp_buff[k].append(v)
        self._tb_idx += 1
        if self._tb_idx == self._sample_size:
            buff = {k: copy.copy(v) for k, v in self._temp_buff.items()}
            for k in ['reward', 'discount']:
                buff[k] = np.array(buff[k], self._dtype)
            for k in self._state_keys:
                buff[k] = buff[k][0]
            self.merge(buff)
            self._data_structure.update(self._mem_idx-1, self._top_priority)
            self._tb_idx = self._burn_in_size

    def merge(self, sequence):
        for k, v in sequence.items():
            if k in self._state_keys:
                np.testing.assert_equal(len(v.shape), 1)
            else:
                np.testing.assert_equal(len(v), self._sample_size)
        self._memory.append(sequence)
        self._mem_idx = (self._mem_idx + 1) % self._capacity
            
    def clear_temp_buffer(self):
        for k in self._temp_buff:
            self._temp_buff[k].clear()
        self._tb_idx = 0

    def _get_samples(self, idxes):
        results = collections.defaultdict(list)
        [results[k].append(v) for i in idxes for k, v in self._memory[i].items()]
        results = {k: np.stack(v) for k, v in results.items()}

        for k, v in results.items():
            np.testing.assert_equal(v.shape[0], self._batch_size)
        return results

if __name__ == "__main__":
    config = dict(
        type='psr',                      # per or uniform
        precision=32,
        # arguments for PER
        beta0=0.4,
        to_update_top_priority=False,

        # arguments for general replay
        batch_size=2,
        sample_size=5,
        burn_in_size=2,
        min_size=5,
        capacity=100,
    )
    replay = SequentialPER(config, state_keys=['h', 'c'])
    for i in range(100):
        h = np.ones(3) * i
        c = np.ones(3) * i
        o = np.ones(2) * i
        replay.add(o=o, h=h, c=c)
    replay.clear_temp_buffer()

    print('sample')
    for k, v in replay.sample().items():
        print(k, v.shape)