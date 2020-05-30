from abc import ABC
import numpy as np

from core.decorator import config
from utility.utils import to_int
from replay.utils import *


class Replay(ABC):
    """ Interface """
    @config
    def __init__(self):
        # params for general replay buffer
        self._min_size = max(self._min_size, self._batch_size*10)
        self._pre_dims = (self._capacity, )
        self._precision = getattr(self, '_precision', 32)

        self._is_full = False
        self._mem_idx = 0
        
        self._memory = {}

    def name(self):
        return self._type

    def good_to_learn(self):
        return len(self) >= self._min_size

    def __len__(self):
        return self._capacity if self._is_full else self._mem_idx

    def __call__(self):
        while True:
            yield self.sample()

    def sample(self, batch_size=None):
        raise NotImplementedError

    def merge(self, local_buffer, length=None, **kwargs):
        """ Merge a local buffer to the replay buffer, 
        useful for distributed algorithms """
        length = length or len(next(iter(local_buffer.values())))
        assert length < self._capacity, (
            f'Local buffer cannot be largeer than the replay: {length} vs. {self._capacity}')
        self._merge(local_buffer, length)

    def add(self, **kwargs):
        """ Add a single transition to the replay buffer """
        next_obs = kwargs['next_obs']
        if self._memory == {}:
            if not self._has_next_obs:
                del kwargs['next_obs']
            init_buffer(self._memory, 
                        pre_dims=self._pre_dims, 
                        has_steps=self._n_steps>1, 
                        precision=self._precision,
                        **kwargs)
            print_buffer(self._memory)

        if not self._is_full and self._mem_idx == self._capacity - 1:
            print(f'Memory is full({len(self._memory["reward"])})')
            self._is_full = True
        
        add_buffer(
            self._memory, self._mem_idx, self._n_steps, self._gamma, cycle=self._is_full, **kwargs)
        self._mem_idx = (self._mem_idx + 1) % self._capacity
        if 'next_obs' not in self._memory:
            self._memory['obs'][self._mem_idx] = next_obs

    """ Implementation """
    def _sample(self, batch_size=None):
        raise NotImplementedError

    def _merge(self, local_buffer, length):
        if self._memory == {}:
            if not self._has_next_obs and 'next_obs' in local_buffer:
                del local_buffer['next_obs']
            init_buffer(self._memory, 
                        pre_dims=self._pre_dims, 
                        has_steps=self._n_steps>1, 
                        precision=self._precision,
                        **local_buffer)
            print_buffer(self._memory)

        end_idx = self._mem_idx + length

        if end_idx > self._capacity:
            first_part = self._capacity - self._mem_idx
            second_part = length - first_part
            
            copy_buffer(self._memory, self._mem_idx, self._capacity, local_buffer, 0, first_part)
            copy_buffer(self._memory, 0, second_part, local_buffer, first_part, length)
        else:
            copy_buffer(self._memory, self._mem_idx, end_idx, local_buffer, 0, length)

        # memory is full, recycle buffer according to FIFO
        if not self._is_full and end_idx >= self._capacity:
            print(f'Memory is full({len(self._memory["reward"])})')
            self._is_full = True
        
        self._mem_idx = end_idx % self._capacity

    def _get_samples(self, idxes):
        """ retrieve samples from replay memory """
        results = {}
        idxes = np.array(idxes, copy=False, dtype=np.int32)
        for k, v in self._memory.items():
            if isinstance(v, np.ndarray):
                results[k] = v[idxes]
            else:
                results[k] = np.array([np.array(v[i], copy=False) for i in idxes])
            
        if 'next_obs' not in self._memory:
            steps = results.get('steps', 1)
            next_idxes = (idxes + steps) % self._capacity
            if isinstance(self._memory['obs'], np.ndarray):
                results['next_obs'] = self._memory['obs'][next_idxes]
            else:
                results['next_obs'] = np.array(
                    [np.array(self._memory['obs'][i], copy=False) 
                    for i in next_idxes])

        return results