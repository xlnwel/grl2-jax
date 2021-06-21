import collections
import time
import threading
import logging
import numpy as np

from utility.utils import batch_dicts, standardize
from replay.utils import *
from algo.ppo.buffer import Buffer as PPOBufffer, \
    compute_indices, compute_gae, compute_nae, \
    reshape_to_sample, reshape_to_store

logger = logging.getLogger(__name__)


class Buffer(PPOBufffer):
    def _add_attributes(self):
        super()._add_attributes()
        self._cache = []
        self._idx = 0

        assert self._n_trajs // self._n_envs * self._n_envs == self._n_trajs, \
            (self._n_trajs, self._n_envs)
        self._n = self._n_trajs // self._n_envs

        # to avoid contention caused by multi-thread parallelism
        self._lock = threading.Lock()

        self._sleep_time = 0.025
        self._sample_wait_time = 0

    def __getitem__(self, k):
        if self._memory is None:
            raise NotImplementedError
        else:
            return self._memory[k]

    def __contains__(self, k):
        if self._memory is None:
            raise NotImplementedError
        else:
            return k in self._memory

    def good_to_learn(self):
        return self._idx >= self._n

    def reset(self):
        self._memory = None
        self._mb_idx = 0
        self._sample_wait_time = 0

    def add(self):
        """ No need for method <add> """
        raise NotImplementedError

    def merge(self, data):
        with self._lock:
            self._cache.append(data)
            self._idx += 1

    def wait_to_sample(self, policy_version):
        while self._idx < self._n:
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time

        with self._lock:
            self._memory = self._cache[-self._n:]
            assert len(self._memory) == self._n, (len(self._memory), self._n)
            self._cache = []
            n = self._idx
            self._idx = 0
        # TODO: do not batch this to avoid copy data from shared memory
        self._memory = batch_dicts(self._memory, np.concatenate)

        self._trajs_dropped = n * self._n_envs - self._n_trajs
        print('Buffer sample wait:', self._sample_wait_time)
        print('#trajs dropped:', self._trajs_dropped)
        # for k, v in self._memory.items():
        #     print(k, v.shape)
        train_step = self._memory["train_step"]
        self._policy_version_min_diff = policy_version - train_step.max()
        self._policy_version_max_diff = policy_version - train_step.min()
        self._policy_version_avg_diff = policy_version - train_step.mean()
        print(f'train_steps: max_diff({self._policy_version_max_diff})',
            f'min_diff({self._policy_version_min_diff})',
            f'avg_diff({self._policy_version_avg_diff})',
            sep='\n')

    def reshape_to_sample(self):
        for k, v in self._memory.items():
            self._memory[k] = np.concatenate(v, 0)

    def sample(self, sample_keys=None):
        assert self._memory is not None

        if self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)

        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, 
            self._mb_size, self.N_MBS)
        
        sample = {k: self._memory[k][self._curr_idxes, 0]
            if k in self._state_keys 
            else self._memory[k][self._curr_idxes] 
            for k in sample_keys}
        
        if self._norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], epsilon=self._epsilon)
        
        return sample

    def get_async_stats(self):
        return {
            'sample_wait_time': self._sample_wait_time,
            'trajs_dropped': self._trajs_dropped,
            'policy_version_min_diff': self._policy_version_min_diff,
            'policy_version_max_diff': self._policy_version_max_diff,
            'policy_version_avg_diff': self._policy_version_avg_diff,
        }


class LocalBuffer(PPOBufffer):
    def is_full(self):
        return self._idx == self.N_STEPS

    def reset(self):
        # assert self.is_full(), self._idx
        self.reshape_to_store()
        self._is_store_shape = True
        self._idx = 0

    def sample(self):
        return self._memory

    def reshape_to_sample(self):
        # do not reshape for sampling. 
        # we maintain the sequential relation
        # making it easier for learner to compute the rms of rewards
        pass

    # methods not supposed to be implemented for LocalBuffer
    def update(self):
        raise NotImplementedError

    def update_value_with_func(self):
        raise NotImplementedError

    def compute_mean_max_std(self, name):
        raise NotImplementedError

    def compute_fraction(self, name):
        raise NotImplementedError
