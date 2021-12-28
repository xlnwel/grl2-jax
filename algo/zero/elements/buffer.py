import time
import logging
import collections
import numpy as np

from algo.ppo.elements.buffer import compute_indices
from core.log import do_logging
from utility.utils import dict2AttrDict, standardize

logger = logging.getLogger(__name__)


def compute_gae(reward, discount, value, last_value, gamma, 
        gae_discount, norm_adv=False, mask=None, epsilon=1e-8):
    last_value = np.expand_dims(last_value, 0)
    next_value = np.concatenate([value[1:], last_value], axis=0)
    assert value.shape == next_value.shape, (value.shape, next_value.shape)
    advs = delta = (reward + discount * gamma * next_value - value).astype(np.float32)
    assert advs.dtype == np.float32, advs.dtype
    next_adv = 0
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] 
            + discount[i] * gae_discount * next_adv)
    traj_ret = advs + value
    if norm_adv:
        advs = standardize(advs, mask=mask, epsilon=epsilon)
    return advs, traj_ret


def get_sample(memory, idxes, sample_keys, state_keys, state_type):
    sample = {k: memory[k][idxes, 0]
        if k in state_keys else memory[k][idxes] 
        for k in sample_keys}
    sample = {}
    state = []
    for k in sample_keys:
        if k in state_keys:
            state.append(memory[k][idxes, 0])
        else:
            sample[k] = memory[k][idxes]
    sample['state'] = state_type(*state)

    return sample


class LocalBuffer:
    def __init__(self, 
                 config: dict2AttrDict, 
                 n_players: int):
        self.config = config

        self._n_envs = self.config.n_envs
        self._gae_discount = self.config.gamma * self.config.lam
        self._maxlen = self.config.N_STEPS
        self._n_players = n_players

        self.reset()

    def size(self):
        return self._memlen

    def reset(self):
        self._memlen = 0
        self._buffers = [collections.defaultdict(list) for _ in range(self._n_envs)]
        self._buff_lens = [0 for _ in range(self._n_envs)]
        self._train_steps = [[None for _ in range(self._n_players)] 
            for _ in range(self._n_envs)]

    def add(self, data):
        for i in range(self._n_envs):
            for k, v in data.items():
                assert v.shape[:2] == (self._n_envs, self._n_players), (k, v.shape, (self._n_envs, self._n_players))
                self._buffers[i][k].append(v[i])
            self._buff_lens[i] += 1
        self._memlen += 1

    def retrieve_all_data(self, last_value):
        assert self._memlen == self._maxlen, (self._memlen, self._maxlen)
        episode = {k: np.stack([np.stack(b[k]) for b in self._buffers], 1) 
            for k in self._buffers[0].keys()}
        for k, v in episode.items():
            assert v.shape[:3] == (self._memlen, self._n_envs, self._n_players), \
                (k, v.shape, (self._memlen, self._n_envs, self._n_players))

        episode['advantage'], episode['traj_ret'] = compute_gae(
            episode['reward'], episode['discount'], episode['value'], 
            last_value, self.config.gamma, self._gae_discount)
        
        data = {k: np.swapaxes(episode[k], 0, 1) 
            for k in self.config.sample_keys}
        for k, v in data.items():
            assert v.shape[:3] == (self._n_envs, self._memlen, self._n_players), \
                (k, v.shape, (self._n_envs, self._memlen, self._n_players))
        
        return data, self._memlen * self._n_envs

    def retrieve_episode(self, eid, last_value):
        last_value = np.zeros((self._n_players,), dtype=np.float32)
        episode = {k: np.stack(v) for k, v in self._buffers[eid].items()}
        for k, v in episode.items():
            assert v.shape[:3] == (self._memlen, self._n_envs, self._n_players), \
                (k, v.shape, (self._memlen, self._n_envs, self._n_players))

        episode['advantage'], episode['traj_ret'] = compute_gae(
            episode['reward'], episode['discount'], episode['value'], 
            last_value, self.config.gamma, self._gae_discount)

        epslen = self._buff_lens[eid]
        self._buffers[eid] = collections.defaultdict(list)
        self._buff_lens[eid] = 0

        return episode, epslen


class PPOBuffer:
    def __init__(self, config, model):
        self.config = dict2AttrDict(config)
        self._norm_adv = getattr(self.config, 'norm_adv', False)
        self._use_dataset = getattr(self.config, 'use_dataset', False)
        if self._use_dataset:
            do_logging(f'Dataset is used for data pipline', logger=logger)

        self._sample_size = self.config.sample_size
        self._sample_keys = self.config.sample_keys
        self._state_keys = model.state_keys
        self._state_type = model.state_type

        self._max_size = self.config.n_workers * self.config.n_envs * self.config.N_STEPS
        self._current_size = 0
        self._batch_size = self._max_size // self._sample_size
        self._mb_size = self._batch_size // self.config.N_MBS
        self._idxes = np.arange(self._batch_size)
        self._shuffled_idxes = np.arange(self._batch_size)
        self._gae_discount = self.config.gamma * self.config.lam
        self._epsilon = 1e-5
        self.reset()
        do_logging(f'Batch size: {self._batch_size}', logger=logger)
        do_logging(f'Mini-batch size: {self._mb_size}', logger=logger)

        self._sleep_time = 0.025
        self._sample_wait_time = 0
        self._epoch_idx = 0
        self._merge_episodes = None # flag to check if calling merge_data and merge_episode in a single run

    def __getitem__(self, k):
        return self._memory[k]

    def __contains__(self, k):
        return k in self._memory
    
    def ready(self):
        return self._ready
    
    def size(self):
        return self._current_size
    
    def max_size(self):
        return self._max_size

    def reset(self):
        self._memory = collections.defaultdict(list)
        self._mb_idx = 0
        self._epoch_idx = 0
        self._current_size = 0
        self._ready = False

    def merge_data(self, data, n):
        if self._merge_episodes == True:
            assert False
        self._merge_episodes = False

        for k, v in data.items():
            self._memory[k].append(v)
        self._current_size += n

        if self._current_size == self._max_size:
            for k, v in self._memory.items():
                v = np.concatenate(v)
                self._memory[k] = np.reshape(v, 
                    (self._batch_size, self._sample_size, *v.shape[2:]))
            self._ready = True

    def merge_episode(self, episode, n):
        assert False, 'not support yet'
        if self._merge_episodes == False:
            assert False
        self._merge_episodes = True

        for k, v in episode.items():
            self._memory[k].append(v)
        self._current_size += n
        print('merge episode', self._current_size, self._max_size)
        if self._current_size > self._max_size:
            self.finish()
            return True
        else:
            return False

    def finish(self):
        for k, v in self._memory.items():
            v = np.concatenate(v)   # merge env and sequential dimensions
            assert v.shape[0] > self._max_size, v.shape
            v = v[:self._max_size]
            self._memory[k] = v.reshape(self._batch_size, self._sample_size, *v.shape[1:])
        self._ready = True

    def sample(self, sample_keys=None):
        self._wait_to_sample()
        self._shuffle_indices()
        sample = self._sample(sample_keys)
        self._post_process_for_dataset()

        return sample

    """ Implementations """
    def _wait_to_sample(self):
        while not self._ready:
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time
        # print(f'PPOBuffer starts sampling: waiting time: {self._sample_wait_time}')
        self._sample_wait_time = 0

    def _shuffle_indices(self):
        if self.config.N_MBS > 1 and self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)

    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, 
            self._mb_size, self.config.N_MBS)

        sample = get_sample(
            self._memory, self._curr_idxes, sample_keys, self._state_keys, 
            self._state_type)
        sample = self._process_sample(sample)

        return sample

    def _process_sample(self, sample):
        if self._norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], epsilon=self._epsilon)
        return sample

    def _post_process_for_dataset(self):
        if self._mb_idx == 0:
            self._epoch_idx += 1
            if self._epoch_idx == self.config.N_EPOCHS:
                # resetting here is especially important 
                # if we use tf.data as sampling is done 
                # in a background thread
                self.reset()


def create_buffer(config, model, **kwargs):
    config = dict2AttrDict(config)
    if config.type == 'ppo' or config.type == 'pbt':
        return PPOBuffer(config, model, **kwargs)
    elif config.type == 'local':
        return LocalBuffer(config, **kwargs)
    else:
        raise ValueError(config.type)
