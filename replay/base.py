from abc import ABC
import threading
import numpy as np

from utility.display import pwc
from utility.utils import to_int
from utility.run_avg import RunningMeanStd
from replay.utils import *


class Replay(ABC):
    """ Interface """
    def __init__(self, config, *keys, state_shape=None):
        """
        Args:
            config: a dict
            keys: the keys of buffer
            state_shape: state_shape is required if 'next_state' is not in keys
                otherwise, error will arise for distributed algorithms
        """
        if 'next_state' not in keys:
            assert state_shape is not None
        # params for general replay buffer
        self._type = config['type']
        self.capacity = to_int(config['capacity'])
        self.min_size = to_int(config['min_size'])
        self.batch_size = config['batch_size']
        self.n_steps = config['n_steps']
        self.gamma = config['gamma']

        # reward hacking
        self.reward_scale = config.get('reward_scale', 1)
        self.reward_clip = config.get('reward_clip')
        self.normalize_reward = config.get('normalize_reward', False)
        if self.normalize_reward:
            self.running_reward_stats = RunningMeanStd()
        print(f'reward hacking: reward scale({self.reward_scale})',
              f'reward_clip({self.reward_clip})',
              f'noramlize_reward({self.normalize_reward})')
        
        self.is_full = False
        self.mem_idx = 0
        
        self.memory = {}
        init_buffer(self.memory, *keys, capacity=self.capacity, state_shape=state_shape)
        pwc(f'buffer keys: {list(self.memory.keys())}', color='cyan')

        # locker used to avoid conflict introduced by tf.data.Dataset
        # self.locker = threading.Lock()

    def buffer_type(self):
        return self._type

    def good_to_learn(self):
        return len(self) >= self.min_size

    def __len__(self):
        return self.capacity if self.is_full else self.mem_idx

    def __call__(self):
        while True:
            yield self.sample()

    def sample(self):
        assert self.good_to_learn, (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self.min_size})')
        # with self.locker:
        samples = self._sample()

        return samples

    def merge(self, local_buffer, length):
        """ Merge a local buffer to the replay buffer, useful for distributed algorithms """
        assert length < self.capacity, (
            f'Local buffer cannot be largeer than the replay: {length} vs. {self.capacity}')
        # with self.locker:
        self._merge(local_buffer, length)

    def add(self, **kwargs):
        """ Add a single transition to the replay buffer """
        # for k, v in kwargs.items():
        #     assert not np.any(np.isnan(v)), f'{k}: {v}'
        # with self.locker:
        if not self.is_full:
            if self.mem_idx == self.capacity - 1:
                pwc(f'Memory is full({len(self.memory["reward"])})', color='blue')
                self.is_full = True
        next_state = kwargs['next_state']
        add_buffer(
            self.memory, self.mem_idx, self.n_steps, self.gamma, cycle=self.is_full, **kwargs)
        self.mem_idx = (self.mem_idx + 1) % self.capacity
        if 'next_state' not in self.memory:
            self.memory['state'][self.mem_idx] = next_state

    """ Implementation """
    def _sample(self):
        raise NotImplementedError

    def _merge(self, local_buffer, length):
        end_idx = self.mem_idx + length

        if end_idx > self.capacity:
            first_part = self.capacity - self.mem_idx
            second_part = length - first_part
            
            copy_buffer(self.memory, self.mem_idx, self.capacity, local_buffer, 0, first_part)
            copy_buffer(self.memory, 0, second_part, local_buffer, first_part, length)
        else:
            copy_buffer(self.memory, self.mem_idx, end_idx, local_buffer, 0, length)
            
        if self.normalize_reward:
            # compute running reward statistics
            self.running_reward_stats.update(local_buffer['reward'][:length])

        # memory is full, recycle buffer via FIFO
        if not self.is_full and end_idx >= self.capacity:
            pwc(f'Memory is full({len(self.memory["reward"])})', color='blue')
            self.is_full = True
        
        self.mem_idx = end_idx % self.capacity

    def _get_samples(self, indexes):
        # original_state = [self.memory['state'][i] for i in indexes]
        state = np.asarray([np.array(self.memory['state'][i], copy=False) for i in indexes])
        action = np.asarray([np.array(self.memory['action'][i], copy=False) for i in indexes])
        reward = np.asarray([self.memory['reward'][i] for i in indexes])
        done = np.asarray([self.memory['done'][i] for i in indexes])
        steps = np.asarray([self.memory['steps'][i] for i in indexes], dtype=np.int32)
        
        if 'next_state' in self.memory:
            next_state = np.asarray([np.array(self.memory['next_state'][i], copy=False) for i in indexes])
        else:
            indexes = np.asarray(indexes)
            next_indexes = (indexes + steps) % self.capacity
            assert indexes.shape == next_indexes.shape == (self.batch_size, )
            next_state = np.asarray([self.memory['state'][i] for i in next_indexes])

        # process rewards
        if self.normalize_reward:
            reward = self.running_reward_stats.normalize(reward)
        if self.reward_scale != 1:
            reward *= np.where(done, 1, self.reward_scale)
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        # assert not np.any(np.isnan(state))
        # assert not np.any(np.isnan(action))
        # assert not np.any(np.isnan(reward))
        # assert not np.any(np.isnan(next_state))
        # assert not np.any(np.isnan(done))
        # assert not np.any(np.isnan(steps))
        return state, action, reward, next_state, done, steps
