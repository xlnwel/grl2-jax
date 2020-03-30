import numpy as np
from copy import deepcopy

from core.decorator import config
from utility.display import pwc
from utility.utils import moments, standardize
from replay.utils import init_buffer, print_buffer


class PPOBuffer:
    @config
    def __init__(self):
        self._mb_len = self._n_steps // self._n_mbs

        self.gae_discount = self._gamma * self._lam
        
        self._memory = {}
        self.reset()

    def add(self, **data):
        if self._memory == {}:
            init_buffer(self._memory, pre_dims=(self._n_envs, self._n_steps), **data)
            self._memory['value'] = np.zeros((self._n_envs, self._n_steps+1), dtype=np.float32)
            self._memory['traj_ret'] = np.zeros((self._n_envs, self._n_steps), dtype=np.float32)
            self._memory['advantage'] = np.zeros((self._n_envs, self._n_steps), dtype=np.float32)
            print_buffer(self._memory)
            
        for k, v in data.items():
            self._memory[k][:, self._idx] = v

        self._idx += 1

    def sample(self):
        assert self._ready
        start = self._batch_idx * self._mb_len
        end = np.minimum((self._batch_idx + 1) * self._mb_len, self._idx)
        if start > self._idx or (end == self._idx and np.sum(self._memory['mask'][:, start:end]) < 500):
            self._batch_idx = 0
            start = self._batch_idx * self._mb_len
            end = np.minimum((self._batch_idx + 1) * self._mb_len, self._idx)
        else:
            self._batch_idx = (self._batch_idx + 1) % self._n_mbs

        keys = ['obs', 'action', 'traj_ret', 'value', 
                'advantage', 'old_logpi', 'mask']

        return {k: self._memory[k][:, start:end] for k in keys}

    def finish(self, last_value):
        self._memory['value'][:, self._idx] = last_value
        valid_slice = np.s_[:, :self._idx]
        self._memory['mask'][:, self._idx:] = 0
        mask = self._memory['mask'][valid_slice]

        # Environment hack
        if hasattr(self, '_reward_scale'):
            self._memory['reward'] *= self._reward_scale
        if hasattr(self, '_reward_clip'):
            self._memory['reward'] = np.clip(self._memory['reward'], -self.reward_clip, self.reward_clip)

        if self._adv_type == 'nae':
            traj_ret = self._memory['traj_ret'][valid_slice]
            next_return = last_value
            for i in reversed(range(self._idx)):
                traj_ret[:, i] = next_return = (self._memory['reward'][:, i] 
                    + self._memory['nonterminal'][:, i] * self._gamma * next_return)

            # Standardize traj_ret and advantages
            traj_ret_mean, traj_ret_std = moments(traj_ret, mask=mask)
            value = standardize(self._memory['value'][valid_slice], mask=mask)
            # To have the same mean and std as trajectory return
            value = (value + traj_ret_mean) / (traj_ret_std + 1e-8)     
            self._memory['advantage'][valid_slice] = standardize(traj_ret - value, mask=mask)
            self._memory['traj_ret'][valid_slice] = standardize(traj_ret, mask=mask)
        elif self._adv_type == 'gae':
            advs = delta = (self._memory['reward'][valid_slice] 
                + self._memory['nonterminal'][valid_slice] 
                * self._gamma * self._memory['value'][:, 1:self._idx+1]
                - self._memory['value'][valid_slice])
            next_adv = 0
            for i in reversed(range(self._idx)):
                advs[:, i] = next_adv = (delta[:, i] 
                + self._memory['nonterminal'][:, i] * self.gae_discount * next_adv)
            self._memory['traj_ret'][valid_slice] = advs + self._memory['value'][valid_slice]
            self._memory['advantage'][valid_slice] = standardize(advs, mask=mask)
        else:
            raise NotImplementedError

        for k, v in self._memory.items():
            shape = v[valid_slice].shape
            v[valid_slice] = np.reshape((v[valid_slice].T * mask.T).T, shape)
        
        self._ready = True

    def reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._ready = False      # Whether the buffer is ready to be read


if __name__ == '__main__':
    gamma = .99
    lam = .95
    gae_discount = gamma * lam
    config = dict(
        gamma=gamma,
        lam=lam,
        advantage_type='gae',
        _n_mbs=2
    )
    kwargs = dict(
        config=config,
        n_envs=8, 
        seqlen=1000, 
        _n_mbs=2, 
    )
    buffer = PPOBuffer(**kwargs)
    d = np.zeros((kwargs['n_envs']))
    m = np.ones((kwargs['n_envs']))
    for i in range(kwargs['seqlen']):
        r = np.random.rand(kwargs['n_envs'])
        v = np.random.rand(kwargs['n_envs'])
        if np.random.randint(2):
            d[np.random.randint(kwargs['n_envs'])] = 1
        buffer.add(reward=r,
                value=v,
                nonterminal=1-d,
                mask=m)
        m = 1-d
        if np.all(d == 1):
            break
    last_value = np.random.rand(kwargs['n_envs'])
    buffer.finish(last_value)
    