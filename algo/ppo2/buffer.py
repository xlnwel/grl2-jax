import numpy as np
import tensorflow as tf

from core.decorator import config
from utility.display import pwc
from utility.utils import moments, standardize
from replay.utils import init_buffer, print_buffer


class PPOBuffer:
    @config
    def __init__(self):
        self._mb_len = self._n_envs // self._n_mbs
        self._env_ids = np.arange(self._n_envs)

        self._gae_discount = self._gamma * self._lam

        self._memory = {}
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

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
        if self._mb_idx == 0:
            np.random.shuffle(self._env_ids)
        start = self._mb_idx * self._mb_len
        end = np.minimum((self._mb_idx + 1) * self._mb_len, self._idx)
        env_ids = self._env_ids[start: end]
        self._mb_idx = (self._mb_idx + 1) % self._n_mbs

        keys = ['obs', 'action', 'traj_ret', 'value', 
                'advantage', 'old_logpi', 'mask']
                
        sample = {k: self._memory[k][env_ids, :self._n_steps] for k in keys}
        sample['h'] = self.state[0][env_ids]
        sample['c'] = self.state[1][env_ids]

        return sample

    def finish(self, last_value):
        self._memory['value'][:, self._idx] = last_value
        valid_slice = np.s_[:, :self._idx]
        self._memory['mask'][:, self._idx:] = 0
        mask = self._memory['mask'][valid_slice]

        # Environment hack
        if hasattr(self, '_reward_scale'):
            self._memory['reward'] *= self._reward_scale
        if hasattr(self, '_reward_clip'):
            self._memory['reward'] = np.clip(self._memory['reward'], -self._reward_clip, self._reward_clip)

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
                + self._memory['nonterminal'][valid_slice] * self._gamma 
                * self._memory['value'][:, 1:self._idx+1]
                - self._memory['value'][valid_slice])
            next_adv = 0
            for i in reversed(range(self._idx)):
                advs[:, i] = next_adv = (delta[:, i] 
                + self._memory['nonterminal'][:, i] * self._gae_discount * next_adv)
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
        self._mb_idx = 0
        self._ready = False
        self._memory['mask'] = np.zeros((self._n_envs, self._n_steps), dtype=bool)

    def store_state(self, state):
        self.state = tf.nest.map_structure(lambda x: x.numpy(), state)
