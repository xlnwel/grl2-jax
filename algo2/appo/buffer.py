import time
import collections
import threading
import logging
import numpy as np

from core.decorator import config
from utility.utils import moments, standardize, expand_dims_match
from replay.utils import init_buffer, print_buffer
from algo.ppo.buffer import compute_gae

logger = logging.getLogger(__name__)


class Buffer:
    @config
    def __init__(self):
        self._add_attributes()

    def _add_attributes(self):
        self._sample_size = getattr(self, '_sample_size', None)
        self._state_keys = getattr(self, '_state_keys', [])
        if self._sample_size:
            assert self._n_envs * self.N_STEPS % self._sample_size == 0, \
                f'{self._n_envs} * {self.N_STEPS} % {self._sample_size} != 0'
            size = self._n_envs * self.N_STEPS // self._sample_size
            logger.info(f'Sample size: {self._sample_size}')
        else:
            size = self._n_envs * self.N_STEPS
        self._size = size
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._shuffled_idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._memory = []
        self._lock = threading.Lock()
        self._is_store_shape = True
        self._inferred_sample_keys = False
        self._norm_adv = getattr(self, '_norm_adv', 'minibatch')
        self._epsilon = 1e-5
        self.sample_wait_time = 0
        self.reset()
        logger.info(f'Batch size: {size}')
        logger.info(f'Mini-batch size: {self._mb_size}')

    def merge(self, data):
        if self._target_type == 'gae':
            data['advantage'], data['traj_ret'] = \
                self._compute_advantage_return(
                    data['reward'], data['discount'], 
                    data['value'], None,
                    epsilon=self._epsilon)
        with self._lock:
            self._memory.append(data)
            self._idx += 1

    def sample(self, policy_version, sample_keys=None):
        while self._idx < self._n_rollouts:
            time.sleep(.01)
            self.sample_wait_time += .01
        with self._lock:
            data = self._memory.copy()
            self._memory.clear()
            n = self._idx
        
        k = sorted(data.keys())[]


        if self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)

        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self.N_MBS)
        
        sample = {k: self._memory[k][self._curr_idxes, 0]
            if k in self._state_keys 
            else self._memory[k][self._curr_idxes] 
            for k in sample_keys}
        
        if self._norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], epsilon=self._epsilon)
        
        return sample

    def _compute_advantage_return(self, reward, discount, value, last_value, 
                                traj_ret=None, mask=None, epsilon=1e-8):
        advantage, traj_ret = compute_gae(
                reward=reward, 
                discount=discount,
                value=value,
                last_value=last_value,
                gamma=self._gamma,
                gae_discount=self._gae_discount,
                norm_adv=self._norm_adv == 'batch',
                mask=mask,
                epsilon=epsilon)

        return advantage, traj_ret
