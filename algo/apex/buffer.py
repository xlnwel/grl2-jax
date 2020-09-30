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
    def add_data(self, obs, action, reward, discount, next_obs):
        raise NotImplementedError


class EnvBuffer(LocalBuffer):
    """ Local memory only stores one episode of transitions from each of n environments """
    @config
    def __init__(self):
        self._memory = {}
        self._idx = 0

    def is_full(self):
        return self._idx == self._seqlen + self._n_steps

    def reset(self):
        self._idx = self._n_steps
        for k, v in self._memory.items():
            v[:self._n_steps] = v[self._seqlen:]

    def add_data(self, **data):
        """ Add experience to local memory """
        next_obs = data['next_obs']
        if self._memory == {}:
            del data['next_obs']
            init_buffer(self._memory, pre_dims=self._seqlen+self._n_steps, has_steps=self._n_steps>1, **data)
            print_buffer(self._memory, 'Local')
            
        add_buffer(self._memory, self._idx, self._n_steps, self._gamma, **data)
        self._idx = self._idx + 1
        self._memory['obs'][self._idx] = next_obs

    def sample(self):
        results = {}
        for k, v in self._memory.items():
            results[k] = v[:self._seqlen]
        if 'next_obs' not in self._memory:
            idxes = np.arange(self._seqlen)
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

    def is_full(self):
        return self._idx == self._seqlen + self._n_steps
        
    def reset(self):
        self._idx = self._n_steps
        for k, v in self._memory.items():
            v[:, :self._n_steps] = v[:, self._seqlen:]

    def add_data(self, env_ids=None, **data):
        """ Add experience to local memory """
        if self._memory == {}:
            # initialize memory
            init_buffer(self._memory, pre_dims=(self._n_envs, self._seqlen + self._n_steps), 
                        has_steps=self._n_steps>1, **data)
            print_buffer(self._memory, 'Local Buffer')

        env_ids = env_ids or range(self._n_envs)
        idx = self._idx
        # for i, env_id in enumerate(env_ids):
        #     for k, v in data.items():
        #         self._memory[k][env_id, idx] = v[i]

        #     self._memory['steps'][env_id, idx] = 1

        #     # Update previous experience if multi-step is required
        #     for j in range(1, self._n_steps):
        #         k = idx - j
        #         k_disc = self._memory['discount'][i, k]
        #         if not k_disc:
        #             break
        #         self._memory['reward'][i, k] += self._gamma**j * data['reward'][i]
        #         self._memory['discount'][i, k] = data['discount'][i]
        #         self._memory['steps'][i, k] += 1
        #         self._memory['next_obs'][i, k] = data['next_obs'][i]
        for k, v in data.items():
            self._memory[k][:, idx] = v
        self._memory['steps'][:, idx] = 1

        # Update previous experience if multi-step is required
        for i in range(1, self._n_steps):
            k = idx - i
            k_disc = self._memory['discount'][:, k]
            self._memory['reward'][:, k] += self._gamma**i * data['reward'] * k_disc
            self._memory['steps'][:, k] += k_disc.astype(np.uint8)
            self._memory['next_obs'][:, k] = np.where(
                (k_disc==1).reshape(-1, 1, 1, 1), data['next_obs'], self._memory['next_obs'][:, k])
            self._memory['discount'][:, k] = data['discount'] * k_disc

        self._idx = self._idx + 1

    def sample(self):
        results = {}
        for k, v in self._memory.items():
            results[k] = v[:, :self._seqlen].reshape((-1, *v.shape[2:]))
        if 'steps' in results:
            results['steps'] = results['steps'].astype(np.float32)

        return results

if __name__ == '__main__':
    n_envs = 2
    buf = EnvVecBuffer(dict(
        seqlen=10, 
        gamma=1,
        n_envs=n_envs,
        n_steps=3
    ))
    for i in range(10):
        obs = np.ones((n_envs, 2))*i
        next_obs = np.ones((n_envs, 2))*(i+1)
        discount = np.ones(n_envs)
        if i == 2 or i == 7:
            discount[0] = 0
        if i == 4:
            discount[1] = 0
        buf.add_data(obs=obs, reward=np.arange(1, n_envs+1, dtype=np.float32)*i, next_obs=next_obs, discount=discount)
        if buf.is_full():
            buf.reset()
    print(buf._memory['steps'][0, :])