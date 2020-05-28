import numpy as np

from core.decorator import config
from utility.display import pwc
from utility.utils import moments, standardize
from replay.utils import init_buffer, print_buffer


class PPOBuffer:
    @config
    def __init__(self):
        size = self._n_envs * self.N_STEPS
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self.reset()
        print(f'Mini-batch size: {self._mb_size}')


    def add(self, **data):
        if self._memory == {}:
            dtype = {16: np.float16, 32: np.float32}[self._precision]
            init_buffer(self._memory, pre_dims=(self._n_envs, self.N_STEPS), 
                        precision=self._precision, **data)
            self._memory['value'] = np.zeros((self._n_envs, self.N_STEPS+1), dtype=dtype)
            self._memory['traj_ret'] = np.zeros((self._n_envs, self.N_STEPS), dtype=dtype)
            self._memory['advantage'] = np.zeros((self._n_envs, self.N_STEPS), dtype=dtype)
            print_buffer(self._memory)
            
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

        keys = ['obs', 'action', 'traj_ret', 'value', 
                'advantage', 'logpi']
        
        return {k: self._memory[k][self._idxes[start: end]] for k in keys}

    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self._memory['value'][:, -1] = last_value

        if self._adv_type == 'nae':
            traj_ret = self._memory['traj_ret']
            next_return = last_value
            for i in reversed(range(self.N_STEPS)):
                traj_ret[:, i] = next_return = (self._memory['reward'][:, i] 
                    + self._memory['discount'][:, i] * self._gamma * next_return)

            # Standardize traj_ret and advantages
            traj_ret_mean, traj_ret_std = moments(traj_ret)
            value = standardize(self._memory['value'][:, :-1])
            # To have the same mean and std as trajectory return
            value = (value + traj_ret_mean) / (traj_ret_std + 1e-8)     
            self._memory['advantage'] = standardize(traj_ret - value)
            self._memory['traj_ret'] = standardize(traj_ret)
        elif self._adv_type == 'gae':
            advs = delta = (self._memory['reward'] 
                + self._memory['discount'] * self._gamma 
                * self._memory['value'][:, 1:]
                - self._memory['value'][:, :-1])
            next_adv = 0
            for i in reversed(range(self.N_STEPS)):
                advs[:, i] = next_adv = (delta[:, i] 
                    + self._memory['discount'][:, i] * self._gae_discount * next_adv)
            self._memory['traj_ret'] = advs + self._memory['value'][:, :-1]
            self._memory['advantage'] = standardize(advs)
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
