from abc import ABC, abstractmethod
import numpy as np

from core.decorator import config
from replay.utils import *


class LocalBuffer(ABC):
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
    """ Local memory only stores one episode of transitions from each of n environments """
    @config
    def __init__(self):
        self._memory = {}
        self._idx = 0
        self._max_steps = getattr(self, '_max_steps', 0)
        self._extra_len = max(self._n_steps, self._max_steps)
        self._memlen = self._seqlen + self._extra_len

    def is_full(self):
        return self._idx == self._memlen

    def reset(self):
        self._idx = self._extra_len
        for v in self._memory.values():
            v[:self._extra_len] = v[self._seqlen:]

    def add(self, **data):
        """ Add experience to local memory """
        if self._memory == {}:
            del data['next_obs']
            init_buffer(self._memory, pre_dims=self._memlen, has_steps=self._n_steps>1, **data)
            print_buffer(self._memory, 'Local')
            
        add_buffer(self._memory, self._idx, self._n_steps, self._gamma, **data)
        self._idx = self._idx + 1

    def sample(self):
        assert self.is_full(), self._idx
        results = {}
        for k, v in self._memory.items():
            results[k] = v[:self._idx-self._n_steps]
        if 'next_obs' not in self._memory:
            idxes = np.arange(self._idx-self._n_steps)
            steps = results.get('steps', 1)
            next_idxes = idxes + steps
            if isinstance(self._memory['obs'], np.ndarray):
                results['next_obs'] = self._memory['obs'][next_idxes]
            else:
                results['next_obs'] = [np.array(self._memory['obs'][i], copy=False) 
                    for i in next_idxes]
        if 'steps' in results:
            results['steps'] = results['steps'].astype(np.float32)
        return results


class EnvVecBuffer:
    """ Local memory only stores one episode of transitions from n environments """
    @config
    def __init__(self):
        self._memory = {}
        self._idx = 0
        self._max_steps = getattr(self, '_max_steps', 0)
        self._extra_len = max(self._n_steps, self._max_steps)
        self._memlen = self._seqlen + self._extra_len

    def is_full(self):
        return self._idx == self._memlen
        
    def reset(self):
        self._idx = self._extra_len
        for k, v in self._memory.items():
            v[:, :self._extra_len] = v[:, self._seqlen:]

    def add(self, env_ids=None, **data):
        """ Add experience to local memory """
        if self._memory == {}:
            # initialize memory
            init_buffer(self._memory, pre_dims=(self._n_envs, self._memlen), 
                        has_steps=self._extra_len>1, **data)
            print_buffer(self._memory, 'Local Buffer')

        env_ids = env_ids or range(self._n_envs)
        idx = self._idx
        
        for k, v in data.items():
            if isinstance(self._memory[k], np.ndarray):
                self._memory[k][:, idx] = v
            else:
                for i in range(self._n_envs):
                    self._memory[k][i][idx] = v[i]
        self._memory['steps'][:, idx] = 1

        self._idx = self._idx + 1

    def sample(self):
        assert self.is_full(), self._idx
        results = adjust_n_steps_envvec(self._memory, self._seqlen, self._n_steps, self._max_steps, self._gamma)
        for k, v in results.items():
            results[k] = v[:, :self._seqlen].reshape((-1, *v.shape[2:]))
        if 'steps' in results:
            results['steps'] = results['steps'].astype(np.float32)
        if 'mask' in results:
            mask = results.pop('mask')
            results = {k: v[mask] for k, v in results.items()}
        return results
