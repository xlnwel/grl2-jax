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
    traj_ret_mean, traj_ret_std = moments(traj_ret)
    value = standardize(value)
    # To have the same mean and std as trajectory return
    value = (value + traj_ret_mean) / (traj_ret_std + 1e-8)     
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


class PPOBuffer:
    @config
    def __init__(self, **kwargs):
        size = self._n_envs * self.N_STEPS
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self.reset()
        print(f'Mini-batch size: {self._mb_size}')


    def add(self, **data):
        if self._memory == {}:
            init_buffer(self._memory, pre_dims=(self._n_envs, self.N_STEPS), **data)
            self._memory['traj_ret'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            self._memory['advantage'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            print_buffer(self._memory)
            self._sample_keys = set(self._memory.keys()) - set(('discount',))#set(('discount', 'reward'))
            
        for k, v in data.items():
            self._memory[k][:, self._idx] = v

        self._idx += 1

    def sample(self):
        assert self._ready
        if self._mb_idx == 0:
            np.random.shuffle(self._idxes)
        start = self._mb_idx * self._mb_size
        end = (self._mb_idx + 1) * self._mb_size
        self._mb_idx = (self._mb_idx + 1) % self.N_MBS
        
        return {k: self._memory[k][self._idxes[start: end]] for k in self._sample_keys}

    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
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

        for k, v in self._memory.items():
            self._memory[k] = np.reshape(v, (-1, *v.shape[2:]))
        
        self._ready = True

    def reset(self):
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

        self._memory = {
            k: np.reshape(v, (self._n_envs, -1, *v.shape[1:]))
            for k, v in self._memory.items()}
