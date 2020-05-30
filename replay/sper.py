import math
import collections
import numpy as np

from utility.utils import convert_dtype
from replay.per import ProportionalPER


class SequentialPER(ProportionalPER):
    def __init__(self, config, state_keys=[]):
        super().__init__(config)
        self._state_keys = state_keys
        self._temp_buff = {}
        self._memory = collections.deque(maxlen=self._capacity)
        self._pop_size = self._sample_size - self._burn_in_size
        self._tb_idx = 0

    def __len__(self):
        return len(self._memory)

    def pre_add(self, **kwargs):
        """ This function should only be called to add necessary stats
        when the environment is reset """
        for k, v in kwargs.items():
            assert k in ['obs', 'prev_action', 'prev_reward'], k
            if k not in self._temp_buff:
                self._temp_buff[k] = collections.deque(maxlen=self._sample_size+1)
            self._temp_buff[k].append(v)
        
    def add(self, **kwargs):
        assert self._tb_idx < self._sample_size
        for k, v in kwargs.items():
            if k in self._temp_buff:
                pass
            elif k in self._state_keys:
                self._temp_buff[k] = collections.deque(
                    maxlen=math.ceil(self._sample_size / self._pop_size))
            else:
                self._temp_buff[k] = collections.deque(maxlen=self._sample_size)
            if k not in self._state_keys or self._tb_idx % self._pop_size == 0:
                self._temp_buff[k].append(v)

        self._tb_idx += 1
        if self._tb_idx == self._sample_size:
            buff = {k: v[0] if k in self._state_keys 
                    else convert_dtype(v, precision=self._precision)
                    for k, v in self._temp_buff.items()}
            self.merge(buff)
            self._tb_idx = self._burn_in_size

    def merge(self, local_buffer):
        """ Add local_buffer to memory """
        if 'priority' in local_buffer:
            priority = local_buffer['priority']
            del local_buffer['priority']
        else:
            priority = self._top_priority
        np.testing.assert_array_less(0, priority)
        self._data_structure.update(self._mem_idx, priority)
        for k, v in local_buffer.items():
            if k in self._state_keys:
                np.testing.assert_equal(len(v.shape), 1)
            elif k in ['obs', 'prev_action', 'prev_reward']:
                np.testing.assert_equal(len(v), self._sample_size+1)
            else:
                np.testing.assert_equal(len(v), self._sample_size)
        self._memory.append(local_buffer)
        self._mem_idx = (self._mem_idx + 1) % self._capacity
        if not self._is_full and self._mem_idx == 0:
            print(f'Memory is full({len(self)})')
            self._is_full = True

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
