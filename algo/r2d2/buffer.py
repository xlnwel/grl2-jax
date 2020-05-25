import math
import collections
from abc import ABC, abstractmethod
import numpy as np

from core.decorator import config
from replay.utils import *


def create_local_buffer(config, **kwargs):
    buffer_type = EnvBuffer if config.get('n_envs', 1) == 1 else EnvVecBuffer
    return buffer_type(config, **kwargs)


class LocalBuffer(ABC):
    
    def seqlen(self):
        return self._seqlen

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def add(self, **data):
        raise NotImplementedError


class EnvBuffer(LocalBuffer):
    @config
    def __init__(self, state_keys):
        self._memory = collections.defaultdict(list)
        self._state_keys = state_keys
        self._pop_size = self._sample_size - self._burn_in_size
        self._idx = 0

    def is_full(self):
        return self._idx == self._sample_size

    def reset(self):
        for k in self._memory:
            self._memory[k].clear()
        self._idx = 0

    def add(self, **data):
        assert self._idx < self._sample_size
        if self._memory == {}:
            for k in data:
                n_states = math.ceil(self._sample_size / self._pop_size)
                self._memory[k] = collections.deque(
                    maxlen=n_states if k in self._state_keys else self._sample_size)
        else:
            np.testing.assert_equal(set(self._memory), set(data))
        for k, v in data.items():
            if k in self._state_keys:
                if self._idx % self._pop_size == 0:
                    self._memory[k].append(v)
            else:
                self._memory[k].append(v)
        self._idx += 1

    def sample(self):
        data = {k: np.array(v, copy=False) for k, v in self._memory.items()}
        for k in ['reward', 'discount']:
            data[k] = data[k].astype(np.float32)
        for k in self._state_keys:
            data[k] = data[k][0]
        self._idx = self._burn_in_size
        return data


if __name__ == '__main__':
    config = dict(
        type='sper',                      # per or uniform
        precision=32,
        # arguments for PER
        beta0=0.4,
        to_update_top_priority=False,

        # arguments for general replay
        batch_size=2,
        sample_size=7,
        burn_in_size=2,
        min_size=2,
        capacity=100,
    )

    buff = EnvBuffer(config, ['h', 'c'])
    n = np.random.randint(100, 1000)
    for i in range(1000):
        h = np.ones(3) * i
        c = np.ones(3) * i
        r = i
        d = i % n != 0
        buff.add(reward=r, discount=d, h=h, c=c)
        if buff.is_full():
            data = buff.sample()
            print(data)
            np.testing.assert_equal(data['reward'][0], data['h'][0])
            np.testing.assert_equal(data['reward'][0], data['c'][0])
            print(buff._idx)
        if not d: buff.reset()