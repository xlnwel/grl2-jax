import numpy as np
from copy import deepcopy

from core.decorator import config
from utility.display import pwc
from utility.utils import moments, standardize
from replay.utils import init_buffer, print_buffer


class PPOBuffer:
    @config
    def __init__(self):
        self._size = self._n_envs * self._n_steps
        self._mb_size = self._size // self._n_mbs
        self._idxes = np.arange(self._size)
        self._gae_discount = self._gamma * self._lam 
        
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
        if self._batch_idx == 0:
            np.random.shuffle(self._idxes)
        start = self._batch_idx * self._mb_size
        end = (self._batch_idx + 1) * self._mb_size

        self._batch_idx = (self._batch_idx + 1) % self._n_mbs

        keys = ['obs', 'action', 'traj_ret', 'value', 
                'advantage', 'old_logpi']
        
        return {k: self._memory[k][self._idxes[start: end]] for k in keys}

    def finish(self, last_value):
        assert self._idx == self._n_steps, self._idx
        self._memory['value'][:, -1] = last_value

        # Environment hack
        if hasattr(self, '_reward_scale'):
            self._memory['reward'] *= self._reward_scale
        if hasattr(self, '_reward_clip'):
            self._memory['reward'] = np.clip(self._memory['reward'], -self._reward_clip, self._reward_clip)

        if self._adv_type == 'nae':
            traj_ret = self._memory['traj_ret']
            next_return = last_value
            for i in reversed(range(self._n_steps)):
                traj_ret[:, i] = next_return = (self._memory['reward'][:, i] 
                    + self._memory['nonterminal'][:, i] * self._gamma * next_return)

            # Standardize traj_ret and advantages
            traj_ret_mean, traj_ret_std = moments(traj_ret)
            value = standardize(self._memory['value'][:, :-1])
            # To have the same mean and std as trajectory return
            value = (value + traj_ret_mean) / (traj_ret_std + 1e-8)     
            self._memory['advantage'] = standardize(traj_ret - value)
            self._memory['traj_ret'] = standardize(traj_ret)
        elif self._adv_type == 'gae':
            advs = delta = (self._memory['reward'] 
                + self._memory['nonterminal'] * self._gamma 
                * self._memory['value'][:, 1:]
                - self._memory['value'][:, :-1])
            next_adv = 0
            for i in reversed(range(self._n_steps)):
                advs[:, i] = next_adv = (delta[:, i] 
                    + self._memory['nonterminal'][:, i] * self._gae_discount * next_adv)
            self._memory['traj_ret'] = advs + self._memory['value'][:, :-1]
            self._memory['advantage'] = standardize(advs)
        else:
            raise NotImplementedError

        for k, v in self._memory.items():
            self._memory[k] = np.reshape(v, (-1, *v.shape[2:]))
        
        self._ready = True

    def reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._ready = False      # Whether the buffer is _ready to be read
        for k, v in self._memory.items():
            if k == 'value':
                self._memory[k] = np.reshape(v, (self._n_envs, self._n_steps+1, *v.shape[1:]))
            else:
                self._memory[k] = np.reshape(v, (self._n_envs, self._n_steps, *v.shape[1:]))

if __name__ == '__main__':
    gamma = .99
    lam = .95
    _gae_discount = gamma * lam
    config = dict(
        gamma=gamma,
        lam=lam,
        _adv_type='gae',
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
    