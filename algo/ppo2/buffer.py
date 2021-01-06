import logging
import numpy as np

from core.decorator import config
from replay.utils import init_buffer, print_buffer
from algo.ppo.buffer import compute_indices, compute_gae, compute_nae


logger = logging.getLogger(__name__)

class Buffer:
    @config
    def __init__(self, state_keys):
        assert self._n_envs * self.N_STEPS % self._sample_size == 0, \
            f'{self._n_envs} * {self.N_STEPS} % {self._sample_size} != 0'
        size = self._n_envs * self.N_STEPS // self._sample_size
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._shuffled_idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self._state_keys = state_keys
        self.reset()
        logger.info(f'Batch size: {size} chunks of {self._sample_size} timesteps')
        logger.info(f'Mini-batch size: {self._mb_size} chunks of {self._sample_size} timesteps')

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
            state = (self._memory[k][curr_idxes, 0] for k in self._state_keys)
            mask = self._memory['mask'][curr_idxes]
            value, state = fn(obs, state=state, mask=mask, return_state=True)
            self.update('value', value, mb_idxes=curr_idxes)
        
        assert mb_idx == 0, mb_idx

    def sample(self):
        assert self._ready
        if self._shuffle and self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self.N_MBS)

        sample = {k: self._memory[k][self._curr_idxes, 0]
            if k in self._state_keys 
            else self._memory[k][self._curr_idxes] 
            for k in self._sample_keys}
        
        return sample

    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self.reshape_to_store()
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

        self.reshape_to_sample()
        self._ready = True

    def reset(self):
        self._idx = 0
        self._mb_idx = 0
        self._ready = False
        self.reshape_to_store()
    
    def reshape_to_sample(self):
        for k, v in self._memory.items():
            self._memory[k] = np.reshape(v, (-1, self._sample_size, *v.shape[2:]))
    
    def reshape_to_store(self):
        self._memory = {
            k: np.reshape(v, (self._n_envs, self.N_STEPS, *v.shape[2:]))
            for k, v in self._memory.items()}
