from abc import ABC
import numpy as np

from core.decorator import config
from utility.utils import to_int
from utility.run_avg import RunningMeanStd
from replay.utils import *


class Replay(ABC):
    """ Interface """
    @config
    def __init__(self):
        # params for general replay buffer
        self._capacity = to_int(self._capacity)
        self._min_size = to_int(max(self._min_size, self._batch_size*10))
        self._pre_dims = (self._capacity, )
        self._precision = getattr(self, '_precision', 32)

        # reward hacking
        if hasattr(self, '_normalize_reward'):
            self._running_reward_stats = RunningMeanStd()
        print(f'reward hacking: reward scale({getattr(self, "_reward_scale", 1)})',
              f'reward_clip({getattr(self, "_reward_clip", None)})',
              f'noramlize_reward({getattr(self, "_normalize_reward", None)})')
        
        self._is_full = False
        self._mem_idx = 0
        
        self._memory = {}

    def buffer_type(self):
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

    def merge(self, local_buffer, length, **kwargs):
        """ Merge a local buffer to the replay buffer, 
        useful for distributed algorithms """
        assert length < self._capacity, (
            f'Local buffer cannot be largeer than the replay: {length} vs. {self._capacity}')
        self._merge(local_buffer, length)

    def add(self, **kwargs):
        """ Add a single transition to the replay buffer """
        nth_obs = kwargs['nth_obs']
        if self._memory == {}:
            if not self._has_next_obs:
                del kwargs['nth_obs']
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
        if 'nth_obs' not in self._memory:
            self._memory['obs'][self._mem_idx] = nth_obs

    """ Implementation """
    def _sample(self, batch_size=None):
        raise NotImplementedError

    def _merge(self, local_buffer, length):
        if self._memory == {}:
            if not self._has_next_obs:
                del local_buffer['nth_obs']
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

        # memory is full, recycle buffer via FIFO
        if not self._is_full and end_idx >= self._capacity:
            print(f'Memory is full({len(self._memory["reward"])})')
            self._is_full = True
        
        self._mem_idx = end_idx % self._capacity

    def _get_samples(self, indexes):
        """ retrieve samples from replay memory """
        results = {}
        indexes = np.array(indexes, copy=False, dtype=np.int32)
        for k, v in self._memory.items():
            if isinstance(v, np.ndarray):
                results[k] = v[indexes]
            else:
                results[k] = np.array([np.array(v[i], copy=False) for i in indexes])
            
        if 'nth_obs' not in self._memory:
            steps = results.get('steps', 1)
            next_indexes = (indexes + steps) % self._capacity
            if isinstance(self._memory['obs'], np.ndarray):
                results['nth_obs'] = self._memory['obs'][next_indexes]
            else:
                results['nth_obs'] = np.array(
                    [np.array(self._memory['obs'][i], copy=False) 
                    for i in next_indexes])

        # process rewards
        if getattr(self, '_reward_scale', 1) != 1:
            results['reward'] *= np.where(results['done'], 1, self._reward_scale)
        if getattr(self, '_reward_clip', None):
            results['reward'] = np.clip(results['reward'], -self._reward_clip, self._reward_clip)
        if getattr(self, '_normalize_reward', None):
            # we update running reward statistics at sampling time
            # since this is when the rewards contribute to the learning process
            self._running_reward_stats.update(results['reward'])
            results['reward'] = self._running_reward_stats.normalize(results['reward'])

        return results