import numpy as np

from core.decorator import config
from replay.utils import init_buffer, print_buffer
from algo.ppo.buffer import compute_gae


class Buffer:
    @config
    def __init__(self):
        size = self._n_envs * self.N_STEPS
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._gae_discount_int = self._gamma_int * self._lam
        self._gae_discount_ext = self._gamma_ext * self._lam
        self._memory = {}
        self.reset()
        print(f'Mini-batch size: {self._mb_size}')

    def __getitem__(self, k):
        return self._memory[k]

    def add(self, **data):
        if self._memory == {}:
            init_buffer(self._memory, pre_dims=(self._n_envs, self.N_STEPS), **data)
            self._memory['discount_int'] = np.ones_like(self._memory['discount'])   # non-episodic
            norm_obs_shape = self._memory['obs'].shape[:-1] + (1, )
            self._memory['obs_norm'] = np.zeros(norm_obs_shape, dtype=np.float32)
            self._memory['traj_ret_int'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            self._memory['traj_ret_ext'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            self._memory['advantage'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
            print_buffer(self._memory)
            
        for k, v in data.items():
            self._memory[k][:, self._idx] = v

        self._idx += 1

    def update(self, key, value, field='mb', mb_idxes=None):
        if field == 'mb':
            mb_idxes = self._curr_idxes if mb_idxes is None else mb_idxes
            self._memory[key][mb_idxes] = value
        elif field == 'all':
            assert self._memory[key].shape == value.shape, (self._memory[key].shape, value.shape)
            self._memory[key] = value
        else:
            raise ValueError(f'Unknown field: {field}. Valid fields: ("all", "mb")')

    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            value_int, value_ext = fn(obs)
            self.update('value_int', value_int, mb_idxes=curr_idxes)
            self.update('value_ext', value_ext, mb_idxes=curr_idxes)
        
        assert mb_idx == 0, mb_idx

    def sample(self):
        assert self._ready
        if self._mb_idx == 0:
            np.random.shuffle(self._idxes)
        start = self._mb_idx * self._mb_size
        end = (self._mb_idx + 1) * self._mb_size
        self._mb_idx = (self._mb_idx + 1) % self.N_MBS

        keys = ['obs', 'obs_norm', 'action', 'traj_ret_int', 'traj_ret_ext', 
            'value_int', 'value_ext', 'advantage', 'logpi']
        
        return {k: self._memory[k][self._idxes[start: end]] for k in keys}

    def get_obs(self, last_obs):
        assert self._idx == self.N_STEPS, self._idx
        return np.concatenate(
            [self._memory['obs'], np.expand_dims(last_obs, 1)], axis=1)

    def finish(self, reward_int, obs_norm, last_value_int, last_value_ext):
        assert self._idx == self.N_STEPS, self._idx
        assert obs_norm.shape == self._memory['obs_norm'].shape, obs_norm.shape
        self._memory['obs_norm'] = obs_norm

        self._memory['traj_ret_int'], adv_int = \
            compute_gae(reward=reward_int, 
                        discount=self._memory['discount_int'],
                        value=self._memory['value_int'],
                        last_value=last_value_int,
                        gamma=self._gamma_int,
                        gae_discount=self._gae_discount_int)
        self._memory['traj_ret_ext'], adv_ext = \
            compute_gae(reward=self._memory['reward'], 
                        discount=self._memory['discount'],
                        value=self._memory['value_ext'],
                        last_value=last_value_ext,
                        gamma=self._gamma_ext,
                        gae_discount=self._gae_discount_ext)

        self._memory['advantage'] = self._int_coef*adv_int + self._ext_coef*adv_ext

        for k, v in self._memory.items():
            assert v.shape[:2] == (self._n_envs, self.N_STEPS)
            self._memory[k] = np.reshape(v, (-1, *v.shape[2:]))
        
        self._ready = True

    def reset(self):
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

        self._memory = {
            k: np.reshape(v, (self._n_envs, -1, *v.shape[1:]))
            for k, v in self._memory.items()}
