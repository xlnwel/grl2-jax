from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np

from core.decorator import config
from replay.utils import *


def create_local_buffer(config):
    buffer_type = EnvBuffer if config.get('n_envs', 1) == 1 else EnvVecBuffer
    return buffer_type(config)


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
    def add_data(self, obs, action, reward, done, nth_obs, mask):
        raise NotImplementedError


class EnvBuffer(LocalBuffer):
    """ Local memory only stores one episode of transitions from each of n environments """
    @config
    def __init__(self):
        self._memory = {}
        self._idx = 0

    def is_full(self):
        return self._idx == self._seqlen

    def reset(self):
        self._idx = 0

    def add_data(self, **data):
        """ Add experience to local memory """
        nth_obs = data['nth_obs']
        if self._memory == {}:
            del data['nth_obs']
            init_buffer(self._memory, pre_dims=self._seqlen+self._n_steps, has_steps=self._n_steps>1, **data)
            # print_buffer(self._memory, 'Local Buffer')
            
        add_buffer(self._memory, self._idx, self._n_steps, self._gamma, **data)
        self._idx = self._idx + 1
        self._memory['obs'][self._idx] = nth_obs

    def sample(self):
        results = {}
        for k, v in self._memory.items():
            results[k] = v[:self._idx]
        if 'nth_obs' not in self._memory:
            idxes = np.arange(self._idx)
            steps = results.get('steps', 1)
            next_idxes = (idxes + steps) % self._capacity
            if isinstance(self._memory['obs'], np.ndarray):
                results['nth_obs'] = self._memory['obs'][next_idxes]
            else:
                results['nth_obs'] = [np.array(self._memory['obs'][i], copy=False) 
                    for i in next_idxes]

        return None, results


class EnvVecBuffer:
    """ Local memory only stores one episode of transitions from n environments """
    @config
    def __init__(self):
        self._memory = {}
        self._idx = 0

    def is_full(self):
        return self._idx == self._seqlen
        
    def reset(self):
        self._idx = 0
        self._memory['mask'] = np.zeros_like(self._memory['mask'], dtype=np.bool)
        
    def add_data(self, env_ids=None, **data):
        """ Add experience to local memory """
        if self._memory == {}:
            # initialize memory
            init_buffer(self._memory, pre_dims=(self._n_envs, self._seqlen + self._n_steps), 
                        has_steps=self._n_steps>1, **data)
            print_buffer(self._memory, 'Local Buffer')

        env_ids = env_ids or range(self._n_envs)
        idx = self._idx
        for i, env_id in enumerate(env_ids):
            for k, v in data.items():
                self._memory[k][env_id, idx] = v[i]
                
                # self._memory[k][env_id, idx] = v[i]
            self._memory['steps'][env_id, idx] = 1

            # Update previous experience if multi-step is required
            for j in range(1, self._n_steps):
                k = idx - j
                k_done = self._memory['done'][i, k]
                if k_done:
                    break
                self._memory['reward'][i, k] += self._gamma**i * data['reward'][i]
                self._memory['done'][i, k] = data['done'][i]
                self._memory['steps'][i, k] += 1
                self._memory['nth_obs'][i, k] = data['nth_obs'][i]

        self._idx = self._idx + 1

    def sample(self):
        results = {}
        mask = self._memory['mask']
        for k, v in self._memory.items():
            if v.dtype == np.object:
                results[k] = np.stack(v[mask])
            elif k == 'mask':
                continue
            else:
                results[k] = v[mask]
            assert results[k].dtype != np.object, f'{k}, {results[k].dtype}'

        return mask, results
