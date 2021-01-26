import numpy as np

from algo.ppo.buffer import compute_gae, reshape_to_store, reshape_to_sample
from algo.ppg.buffer import Replay as ReplayBase
from algo.ppoiqn.buffer import Buffer

class Replay(ReplayBase):
    def __init__(self, config):
        self.N_PI = config['N_PI']
        self._n_segs = config['n_segs']
        assert self.N_PI >= self._n_segs, (self.N_PI, self._n_segs)
        self._n_envs = config['n_envs']
        self.N_STEPS = config['N_STEPS'] * self._n_segs
        buff_size = config['n_envs'] * config['N_STEPS']
        self._size = buff_size * self._n_segs
        self._n_mbs = self._n_segs * config['N_AUX_MBS_PER_SEG']
        self._mb_size = self._size // self._n_mbs
        self._shuffled_idxes = np.arange(self._size)
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

        self._gamma = config['gamma']
        self._gae_discount = config['gamma'] * config['lam']
        self._buff = Buffer(config)
        self._memory = {}

    def aux_finish(self, last_value):
        assert self._idx == 0, self._idx
        self._memory = reshape_to_store(self._memory, self._n_envs, self.N_STEPS)
        reward = np.expand_dims(self._memory['reward'], -1)
        discount = np.expand_dims(self._memory['discount'], -1)
        self._memory['traj_ret'], _ = \
            compute_gae(reward=reward, 
                        discount=discount,
                        value=self._memory['value'],
                        last_value=last_value,
                        gamma=self._gamma,
                        gae_discount=self._gae_discount)

        self._memory = reshape_to_sample(self._memory, self._n_envs, self.N_STEPS)
        self._ready = True