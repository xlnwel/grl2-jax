import math
import collections
from abc import ABC, abstractmethod
import numpy as np

from core.decorator import config
from utility.utils import convert_dtype
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
    def __init__(self, state_keys, **prev_info):
        self._memory = {}
        self._state_keys = state_keys
        self._prev_info = prev_info
        self._trace_size = self._sample_size - self._burn_in_size
        self._idx = 0

    def is_full(self):
        return self._idx == self._sample_size

    def reset(self):
        for k in self._memory:
            self._memory[k].clear()
        self._idx = 0

    def add(self, **kwargs):
        assert self._idx < self._sample_size
        for k, v in kwargs.items():
            if k in self._memory:
                pass
            elif k in self._state_keys:
                self._memory[k] = collections.deque(
                    maxlen=math.ceil(self._sample_size / self._burn_in_size))
            else:
                self._memory[k] = collections.deque(maxlen=self._sample_size)
            if k not in self._state_keys or self._idx % self._burn_in_size == 0:
                self._memory[k].append(v)
        self._idx += 1
        discount = kwargs['discount']
        if discount == 0:
            self.reset()


    def sample(self):
        data = {k: v[0] if k in self._state_keys 
                    else convert_dtype(v, precision=self._precision)
                    for k, v in self._memory.items()}
        self._idx -= self._burn_in_size
        return data
