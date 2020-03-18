from abc import ABC
import numpy as np

from utility.utils import to_int
from utility.run_avg import RunningMeanStd
from replay.utils import *


class Replay(ABC):
    """ Interface """
    def __init__(self, config):
        """
        Args:
            config: a dict
        """
        # params for general replay buffer
        self._type = config['type']
        self._capacity = to_int(config['capacity'])
        self._min_size = max(to_int(config['min_size']), config['batch_size']*10)
        self._batch_size = config['batch_size']
        self._n_steps = config.get('n_steps', 1)
        self._gamma = config['gamma']
        self._has_next_obs = config.get('has_next_obs', False)
        self._pre_dims = (self.capacity, )

        # reward hacking
        self._reward_scale = config.get('reward_scale', 1)
        self._reward_clip = config.get('reward_clip')
        self._normalize_reward = config.get('normalize_reward', False)
        if self._normalize_reward:
            self._running_reward_stats = RunningMeanStd()
        print(f'reward hacking: reward scale({self.reward_scale})',
              f'reward_clip({self.reward_clip})',
              f'noramlize_reward({self.normalize_reward})')
        
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
        assert length < self.capacity, (
            f'Local buffer cannot be largeer than the replay: {length} vs. {self.capacity}')
        self._merge(local_buffer, length)

    def add(self, **kwargs):
        """ Add a single transition to the replay buffer """
        next_obs = kwargs['next_obs']
        if self._memory == {}:
            if not self._has_next_obs:
                del kwargs['next_obs']
            init_buffer(self.memory, pre_dims=self.pre_dims, has_steps=self.n_steps>1, **kwargs)
            print(f"{self.buffer_type()} replay's keys: {list(self._memory.keys())}")

        if not self._is_full and self._mem_idx == self._capacity - 1:
            print(f'Memory is full({len(self.memory["reward"])})')
            self._is_full = True
        
        add_buffer(
            self.memory, self.mem_idx, self.n_steps, self.gamma, cycle=self.is_full, **kwargs)
        self._mem_idx = (self._mem_idx + 1) % self._capacity
        if 'next_obs' not in self._memory:
            self.memory['obs'][self.mem_idx] = next_obs

    """ Implementation """
    def _sample(self, batch_size=None):
        raise NotImplementedError

    def _merge(self, local_buffer, length):
        if self._memory == {}:
            if not self._has_next_obs:
                del local_buffer['next_obs']
            init_buffer(self.memory, pre_dims=self.pre_dims, has_steps=self.n_steps>1, **local_buffer)
            print(f'"{self.buffer_type()}" keys: {list(self._memory.keys())}')

        end_idx = self._mem_idx + length

        if end_idx > self._capacity:
            first_part = self._capacity - self._mem_idx
            second_part = length - first_part
            
            copy_buffer(self.memory, self.mem_idx, self.capacity, local_buffer, 0, first_part)
            copy_buffer(self.memory, 0, second_part, local_buffer, first_part, length)
        else:
            copy_buffer(self.memory, self.mem_idx, end_idx, local_buffer, 0, length)

        # memory is full, recycle buffer via FIFO
        if not self._is_full and end_idx >= self._capacity:
            print(f'Memory is full({len(self.memory["reward"])})')
            self._is_full = True
        
        self._mem_idx = end_idx % self._capacity

    def _get_samples(self, indexes):
        """ retrieve samples from replay memory """
        results = {}
        indexes = np.array(indexes, copy=False, dtype=np.int32)
        for k, v in self._memory.items():
            results[k] = v[indexes]
            
        if 'next_obs' not in self._memory:
            steps = results['steps'] if 'steps' in results else 1
            next_indexes = (indexes + steps) % self._capacity
            results['next_obs'] = self.memory['obs'][next_indexes]

        # process rewards
        if self._reward_scale != 1:
            results['reward'] *= np.where(results['done'], 1, self.reward_scale)
        if self._reward_clip:
            results['reward'] = np.clip(results['reward'], -self.reward_clip, self.reward_clip)
        if self._normalize_reward:
            # we update running reward statistics at sampling time
            # since this is when the rewards contribute to the learning process
            self._running_reward_stats.update(results['reward'])
            results['reward'] = self._running_reward_stats.normalize(results['reward'])

        return results