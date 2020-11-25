from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np

from core.decorator import config
from replay.utils import *


def create_local_buffer(config):
    buffer_type = EnvBuffer if config.get('n_envs', 1) == 1 else EnvVecBuffer
    return buffer_type(config)


def adjust_n_steps(data, seqlen, n_steps, max_steps, gamma):
    results = {}
    for k, v in data.items():
        if k == 'q' or k == 'v':
            vs = v
        else:
            results[k] = v.copy()[:seqlen]
    for i in range(seqlen):
        if n_steps < max_steps:
            for j in range(1, max_steps):
                if results['discount'][i] == 1:
                    cum_rew = results['reward'][i] + gamma**j * data['reward'][i+j]
                    if j >= n_steps and cum_rew + gamma**(j+1) * vs[i+j+1] * data['discount'][i+j+1] \
                        <= results['reward'][i] + gamma**j * vs[i+j] * data['discount'][i+j]:
                        print('break', i, j, cum_rew + gamma**(j+1) * vs[i+j+1] * data['discount'][i+j+1], \
                            results['reward'][i] + gamma**j * vs[i+j] * data['discount'][i+j])
                        break
                    results['reward'][i] = cum_rew
                    results['next_obs'][i] = data['next_obs'][i+j]
                    results['discount'][i] = data['discount'][i+j]
                    results['steps'][i] += 1
                else:
                    break
        else:
            for j in range(1, n_steps):
                if results['discount'][i]:
                    results['reward'][i] = results['reward'][i] * gamma**j * data['reward'][i+j]
                    results['next_obs'][i] = data['next_obs'][i+j]
                    results['discount'][i] = data['discount'][i+j]
                    results['steps'][i] += 1
    return results


def adjust_n_steps_envvec(data, seqlen, n_steps, max_steps, gamma):
    # we do forward update since updating discount in a backward pass is problematic when max_steps > n_steps
    results = {}
    logp = np.zeros_like(data['reward'])
    for k, v in data.items():
        if k == 'q' or k == 'v':
            vs = v
        elif k == 'logp':
            logp = v
        else:
            results[k] = v.copy()[:, :seqlen]
    obs_exp_dims = tuple(range(1, data['obs'].ndim-1))
    for i in range(seqlen):
        cond = np.ones_like(results['reward'][:, 0], dtype=bool)
        if n_steps < max_steps:
            for j in range(1, max_steps):
                disc = results['discount'][:, i]
                jth_rew = data['reward'][:, i+j] - logp[:, i+j]
                cum_rew = results['reward'][:, i] + gamma**j * jth_rew * disc
                cur_cond = disc == 1 if j < n_steps else np.logical_and(
                    disc == 1, cum_rew + gamma**(j+1) * vs[:, i+j+1] * data['discount'][:, i+j+1] \
                        > results['reward'][:, i] + gamma**j * vs[:, i+j] * data['discount'][:, i+j]
                )
                cond = np.logical_and(cond, cur_cond)
                results['reward'][:, i] = np.where(
                    cond, cum_rew, results['reward'][:, i])
                results['next_obs'][:, i] = np.where(
                    np.expand_dims(cond, obs_exp_dims), data['next_obs'][:, i+j], results['next_obs'][:, i])
                results['discount'][:, i] = np.where(
                    cond, data['discount'][:, i+j], results['discount'][:, i])
                results['steps'][:, i] += np.where(
                    cond, np.ones_like(cond, dtype=np.uint8), np.zeros_like(cond, dtype=np.uint8))
        else:
            for j in range(1, n_steps):
                disc = data['discount'][:, i]
                jth_rew = data['reward'][:, i+j] - logp[:, i+j]
                cond = disc == 1
                results['reward'][:, i] = np.where(
                    cond, results['reward'][:, i] + gamma**j * jth_rew * disc, results['reward'][:, i])
                results['next_obs'][:, i] = np.where(
                    np.expand_dims(cond, obs_exp_dims), data['next_obs'][:, i+j], results['next_obs'][:, i])
                results['discount'][:, i] = np.where(
                    cond, data['discount'][:, i+j], results['discount'][:, i])
                results['steps'][:, i] += np.where(
                    cond, 1, 0)
    return results


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
        self._extra_len = max(self._n_steps, self._max_steps)
        self._memlen = self._seqlen + self._extra_len

    def is_full(self):
        return self._idx == self._memlen

    def reset(self):
        self._idx = self._extra_len
        for k, v in self._memory.items():
            v[:self._extra_len] = v[self._seqlen:]

    def add_data(self, **data):
        """ Add experience to local memory """
        next_obs = data['next_obs']
        if self._memory == {}:
            del data['next_obs']
            init_buffer(self._memory, pre_dims=self._memlen, has_steps=self._n_steps>1, **data)
            print_buffer(self._memory, 'Local')
            
        add_buffer(self._memory, self._idx, self._n_steps, self._gamma, **data)
        self._idx = self._idx + 1
        # self._memory['obs'][self._idx] = next_obs

    def sample(self):
        assert self._idx-self._n_steps > 0, self._idx
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

    def add_data(self, env_ids=None, **data):
        """ Add experience to local memory """
        if self._memory == {}:
            # initialize memory
            init_buffer(self._memory, pre_dims=(self._n_envs, self._memlen), 
                        has_steps=self._extra_len>1, **data)
            print_buffer(self._memory, 'Local Buffer')

        env_ids = env_ids or range(self._n_envs)
        idx = self._idx
        
        for k, v in data.items():
            self._memory[k][:, idx] = v
        self._memory['steps'][:, idx] = 1

        # Update previous experience if multi-step is required
        # for i in range(1, self._n_steps):
        #     k = idx - i
        #     if k < 0: break
        #     k_disc = self._memory['discount'][:, k]
        #     self._memory['reward'][:, k] += self._gamma**i * data['reward'] * k_disc
        #     self._memory['steps'][:, k] += k_disc.astype(np.uint8)
        #     self._memory['next_obs'][:, k] = np.where(
        #         (k_disc==1).reshape(-1, 1, 1, 1), data['next_obs'], self._memory['next_obs'][:, k])
        #     self._memory['discount'][:, k] = data['discount'] * k_disc
        # if self._max_steps > self._n_steps:
        #     self._memory['orig_discount'] = data['discount']
        # for i in range(self._n_steps, self._max_steps):
        #     prev_idx = idx - 1
        #     k = prev_idx - i
        #     if k < 0: break
        #     k_disc = self._memory['discount'][:, k]
        #     cum_rew = self._memory['reward'][:, k] + self._gamma**i * k_disc * self._memory['reward'][:, prev_idx]
        #     cond = np.logical_and(k_disc==1, 
        #         cum_rew + self._gamma**(i+1) * k_disc * self._memory['q'][:, idx] \
        #         > self._memory['reward'][:, k] + self._gamma**i * k_disc * self._memory['q'][:, prev_idx])
        #     # we update data at index k based on cond
        #     self._memory['reward'][:, k] = np.where(
        #         cond, cum_rew, self._memory['reward'][:, k])
        #     self._memory['steps'][:, k] += k_disc.astype(np.uint8)
        #     self._memory['next_obs'][:, k] = np.where(
        #         cond.reshape(-1, 1, 1, 1), data['obs'], self._memory['next_obs'][:, k])
        #     self._memory['discount'][:, k] = self._memory['discount'][:, prev_idx] * k_disc
        # assert idx == self._idx
        self._idx = self._idx + 1

    def sample(self):
        results = adjust_n_steps_envvec(self._memory, self._seqlen, self._n_steps, self._max_steps, self._gamma)
        for k, v in results.items():
            results[k] = v[:, :self._seqlen].reshape((-1, *v.shape[2:]))
        if 'steps' in results:
            results['steps'] = results['steps'].astype(np.float32)
        if 'mask' in results:
            mask = results.pop('mask')
            results = {k: v[mask] for k, v in results.items()}
        return results

if __name__ == '__main__':
    n_envs = 2
    buf = EnvBuffer(dict(
        seqlen=10, 
        gamma=1,
        n_steps=3
    ))
    for i in range(15):
        obs = np.ones(2)*i
        next_obs = np.ones(2)*(i+1)
        discount = 0 if i == 7 else 0
        buf.add_data(obs=obs, reward=i, next_obs=next_obs, discount=discount)
        if buf.is_full():
            buf.reset()
    print(buf._memory['obs'])