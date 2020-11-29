import numpy as np
import tensorflow as tf

from core.decorator import config
from utility.display import pwc
from replay.utils import init_buffer, print_buffer
from algo.ppo.buffer import compute_gae, compute_nae

class Buffer:
    @config
    def __init__(self, state_keys):
        assert self._n_envs * self.N_STEPS % self._sample_size == 0, \
            f'{self._n_envs} * {self.N_STEPS} % {self._sample_size} != 0'
        size = self._n_envs * self.N_STEPS // self._sample_size
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self._state_keys = state_keys
        self.reset()
        print(f'Batch size: {size} chunks of {self._sample_size} timesteps')
        print(f'Mini-batch size: {self._mb_size} chunks of {self._sample_size} timesteps')

    def __getitem__(self, k):
        return self._memory[k]
    
    def add(self, **data):
        if self._memory == {}:
            state_dtype = {16: np.float16, 32: np.float32}[self._precision]
            init_buffer(self._memory, pre_dims=(self._n_envs, self.N_STEPS), **data)
            for k in self._state_keys:
                self._memory[k].astype(state_dtype)
            self._memory['traj_ret'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            self._memory['advantage'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            print_buffer(self._memory)
            if not hasattr(self, '_sample_keys'):
                self._sample_keys = set(self._memory.keys()) - set(('discount', 'reward'))
            
        for k, v in data.items():
            self._memory[k][:, self._idx] = v

        self._idx += 1

    def sample(self):
        assert self._ready
        if self._shuffle and self._mb_idx == 0:
            np.random.shuffle(self._idxes)
        start = self._mb_idx * self._mb_size
        end = (self._mb_idx + 1) * self._mb_size
        self._mb_idx = (self._mb_idx + 1) % self.N_MBS

        sample = {k: self._memory[k][self._idxes[start:end]][:, 0] 
            if k in self._state_keys 
            else self._memory[k][self._idxes[start:end]] 
            for k in self._sample_keys}
        
        return sample

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
            self._memory[k] = np.reshape(v, (-1, self._sample_size, *v.shape[2:]))
            
        self._ready = True

    def reset(self):
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

        self._memory = {
            k: np.reshape(v, (self._n_envs, -1, *v.shape[2:]))
            for k, v in self._memory.items()}
