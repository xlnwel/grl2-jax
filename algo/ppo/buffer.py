import numpy as np

from core.decorator import config
from utility.display import pwc
from utility.utils import moments, standardize
from replay.utils import init_buffer, print_buffer


def compute_nae(reward, discount, value, last_value, traj_ret, gamma):
    next_return = last_value
    for i in reversed(range(reward.shape[1])):
        traj_ret[:, i] = next_return = (reward[:, i]
            + discount[:, i] * gamma * next_return)

    # Standardize traj_ret and advantages
    traj_ret_mean, traj_ret_var = moments(traj_ret)
    traj_ret_std = np.maximum(np.sqrt(traj_ret_var), 1e-8)
    value = standardize(value)
    # To have the same mean and std as trajectory return
    value = (value + traj_ret_mean) / traj_ret_std     
    advantage = standardize(traj_ret - value)
    traj_ret = standardize(traj_ret)
    return traj_ret, advantage

def compute_gae(reward, discount, value, last_value, gamma, gae_discount):
    next_value = np.concatenate(
            [value[:, 1:], np.expand_dims(last_value, 1)], axis=1)
    advs = delta = (reward + discount * gamma * next_value - value)
    next_adv = 0
    for i in reversed(range(advs.shape[1])):
        advs[:, i] = next_adv = (delta[:, i] 
            + discount[:, i] * gae_discount * next_adv)
    traj_ret = advs + value
    advantage = standardize(advs)
    return traj_ret, advantage

def compute_indices(idxes, mb_idx, mb_size, N_MBS):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % N_MBS
    curr_idxes = idxes[start: end]
    return mb_idx, curr_idxes

def restore_time_dim(memory, n_envs, n_steps):
    if 'reward' in memory and memory['reward'].shape[:2] != (n_envs, n_steps):
        for k, v in memory.items():
            memory[k] = v.reshape((n_envs, n_steps, *v.shape[1:]))

    return memory


def flatten_time_dim(memory, n_envs, n_steps):
    if 'reward' in memory and memory['reward'].shape[:2] == (n_envs, n_steps):
        for k, v in memory.items():
            memory[k] = v.reshape((-1, *v.shape[2:]))

    return memory


class Buffer:
    @config
    def __init__(self, **kwargs):
        self._size = size = self._n_envs * self.N_STEPS
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._shuffled_idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self.reset()
        print(f'Batch size: {size}')
        print(f'Mini-batch size: {self._mb_size}')

    def __getitem__(self, k):
        return self._memory[k]

    def __contains__(self, k):
        return k in self._memory
    
    def ready(self):
        return self._ready

    def add(self, **data):
        if self._memory == {}:
            init_buffer(self._memory, pre_dims=(self._n_envs, self.N_STEPS), **data)
            self._memory['traj_ret'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            self._memory['advantage'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            print_buffer(self._memory)
            if getattr(self, '_sample_keys', None) is None:
                self._sample_keys = set(self._memory.keys()) - set(('discount', 'reward'))
            
        for k, v in data.items():
            self._memory[k][:, self._idx] = v

        self._idx += 1

    def update(self, key, value, field='mb', mb_idxes=None):
        if field == 'mb':
            mb_idxes = self._curr_idxes if mb_idxes is None else mb_idxes
            self._memory[key][mb_idxes] = value
        elif field == 'all':
            assert self._memory[key].shape == value.shape, (self._memory[key].shape, value.shape)
            self._memory[key] == value
        else:
            raise ValueError(f'Unknown field: {field}. Valid fields: ("all", "mb")')

    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            value = fn(obs)
            self.update('value', value, mb_idxes=curr_idxes)
        
        assert mb_idx == 0, mb_idx

    def sample(self):
        assert self._ready
        if self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self.N_MBS)
        return {k: self._memory[k][self._curr_idxes] for k in self._sample_keys}

    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self.restore_time_dim()
        if self._adv_type == 'nae':
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_nae(reward=self._memory['reward'], 
                            discount=self._memory['discount'],
                            value=self._memory['value'],
                            last_value=last_value,
                            traj_ret=self._memory['traj_ret'],
                            gamma=self._gamma)
        elif self._adv_type == 'gae':
            self._memory['traj_ret'], self._memory['advantage'] = \
                compute_gae(reward=self._memory['reward'], 
                            discount=self._memory['discount'],
                            value=self._memory['value'],
                            last_value=last_value,
                            gamma=self._gamma,
                            gae_discount=self._gae_discount)
        else:
            raise NotImplementedError

        self.flatten_time_dim()
        self._ready = True

    def reset(self):
        self._idx = 0
        self._mb_idx = 0
        self._ready = False
        self.restore_time_dim()

    def restore_time_dim(self):
        self._memory = restore_time_dim(self._memory, self._n_envs, self.N_STEPS)

    def flatten_time_dim(self):
        self._memory = flatten_time_dim(self._memory, self._n_envs, self.N_STEPS)

    def clear(self):
        self._memory = {}
        self.reset()
