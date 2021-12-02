import collections
import random
import time
import logging
import numpy as np


from core.log import do_logging
from utility.utils import batch_dicts, config_attr, dict2AttrDict, standardize


logger = logging.getLogger(__name__)


def compute_gae(reward, discount, value, gamma, gae_discount):
    next_value = np.concatenate([value[1:], [0]], axis=0)
    assert value.shape == next_value.shape, (value.shape, next_value.shape)
    assert discount[-1] == 0, discount
    advs = delta = (reward + discount * gamma * next_value - value)
    next_adv = 0
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] 
            + discount[i] * gae_discount * next_adv)
    traj_ret = advs + value

    return advs, traj_ret


def compute_indices(idxes, mb_idx, mb_size, N_MBS):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % N_MBS
    curr_idxes = idxes[start: end]
    return mb_idx, curr_idxes



class LocalBuffer:
    def __init__(self, config):
        self.config = dict2AttrDict(config)
        self._gae_discount = self.config.gamma * self.config.lam
        self._maxlen = self.config.n_envs * self.config.N_STEPS
        self.reset()

    def reset(self):
        self._memory = []
        self._buffer = collections.defaultdict(lambda: collections.defaultdict(list))
        self._memlen = 0
        self._buff_lens = collections.defaultdict(int)

    def add(self, data):
        eids = data.pop('eid')
        pids = data.pop('pid')
        reward = data['reward']
        discount = data['discount']
        for i, (eid, pid, r, d) in enumerate(zip(eids, pids, reward, discount)):
            assert r.shape == (4,), r.shape
            for k, vs in data.items():
                if k == 'reward':
                    self._buffer[(eid, pid)][k].append(vs[i][pid])
                else:
                    self._buffer[(eid, pid)][k].append(vs[i])
            self._buff_lens[(eid, pid)] += 1
            if d == 0:
                for p in self.config.control_pids:
                    assert self._buff_lens[(eid, p)] > 0, self._buff_lens[(eid, p)]
                    if p != pid:
                        assert self._buffer[(eid, p)]['reward'][-1] == 0, self._buffer[(eid, p)]['reward'][-1]
                        assert self._buffer[(eid, p)]['discount'][-1] == 1, self._buffer[(eid, p)]['discount'][-1]
                        self._buffer[(eid, p)]['reward'][-1] = r[p]
                        self._buffer[(eid, p)]['discount'][-1] = 0
                    assert self._buffer[(eid, p)]['reward'][-1] == r[p], self._buffer[(eid, p)]['reward'][-1]
                    assert self._buffer[(eid, p)]['discount'][-1] == 0, self._buffer[(eid, p)]['discount'][-1]
                    self.merge_episode(self._buffer[(eid, p)], self._buff_lens[(eid, p)])
                    self._buffer[(eid, p)] = collections.defaultdict(list)
                    self._buff_lens[(eid, p)] = 0

    def merge_episode(self, episode, epslen):
        episode = {k: np.stack(v) for k, v in episode.items()}
        for v in episode.values():
            assert v.shape[0] == epslen, (v.shape, epslen)
        episode['advantage'], episode['traj_ret'] = compute_gae(
            episode['reward'], episode['discount'], episode['value'], 
            self.config.gamma, self._gae_discount
        )
        episode = {k: episode[k] for k in self.config.sample_keys}
        self._memory.append(episode)
        self._memlen += epslen

    def ready_to_retrieve(self):
        return self._memlen > self._maxlen

    def retrieve_data(self):
        data = batch_dicts(self._memory, np.concatenate)
        for v in data.values():
            assert v.shape[0] == self._memlen, (v.shape, self._memlen)
        self.reset()
        return data


class PPOBuffer:
    def __init__(self, config):
        self.config = config_attr(self, config)
        self._use_dataset = getattr(self, '_use_dataset', False)
        if self._use_dataset:
            do_logging(f'Dataset is used for data pipline', logger=logger)

        self._sample_size = getattr(self, '_sample_size', None)
        self._state_keys = ['h', 'c']
        assert self._n_envs * self.N_STEPS % self._sample_size == 0, \
            f'{self._n_envs} * {self.N_STEPS} % {self._sample_size} != 0'
        size = self._n_envs * self.N_STEPS // self._sample_size
        do_logging(f'Sample size: {self._sample_size}', logger=logger)

        self._size = size
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._shuffled_idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._epsilon = 1e-5
        if hasattr(self, 'N_VALUE_EPOCHS'):
            self.N_EPOCHS += self.N_VALUE_EPOCHS
        self.reset()
        do_logging(f'Batch size: {size}', logger=logger)
        do_logging(f'Mini-batch size: {self._mb_size}', logger=logger)

        self._sleep_time = 0.025
        self._sample_wait_time = 0
        self._epoch_idx = 0

    @property
    def batch_size(self):
        return self._mb_size

    def __getitem__(self, k):
        return self._memory[k]

    def __contains__(self, k):
        return k in self._memory
    
    def ready(self):
        return self._ready

    def reset(self):
        self._memory = collections.defaultdict(list)
        self._mb_idx = 0
        self._ready = False

    def append_data(self, data):
        for k, v in data.items():
            self._memory[k].append(v)

    def sample(self, sample_keys=None):
        self._wait_to_sample()

        self._shuffle_indices()
        sample = self._sample(sample_keys)
        self._post_process_for_dataset()

        return sample

    def finish(self):
        for k, v in self._memory.items():
            v = np.concatenate(v)[:self._size * self._sample_size]
            self._memory[k] = v.reshape(self._size, self._sample_size, *v.shape[1:])
        self._ready = True

    """ Implementations """
    def _wait_to_sample(self):
        while not self._ready:
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time

    def _shuffle_indices(self):
        if self.N_MBS > 1 and self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        
    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, 
            self._mb_size, self.N_MBS)

        sample = self._get_sample(sample_keys, self._curr_idxes)
        sample = self._process_sample(sample)

        return sample

    def _get_sample(self, sample_keys, idxes):
        sample = {k: self._memory[k][idxes, 0]
            if k in self._state_keys else self._memory[k][idxes] 
            for k in sample_keys}
        action_rnn_dim = sample['action_h'].shape[-1]
        sample['action_h'] = sample['action_h'].reshape(-1, action_rnn_dim)
        sample['action_c'] = sample['action_c'].reshape(-1, action_rnn_dim)
        return sample

    def _process_sample(self, sample):
        if self._norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], epsilon=self._epsilon)
        return sample
    
    def _post_process_for_dataset(self):
        if self._mb_idx == 0:
            self._epoch_idx += 1
            if self._epoch_idx == self.N_EPOCHS:
                # resetting here is especially important 
                # if we use tf.data as sampling is done 
                # in a background thread
                self.reset()


def create_buffer(config, central_buffer=False):
    config = dict2AttrDict(config)
    if central_buffer:
        assert config.type == 'ppo', config.type
        import ray
        RemoteBuffer = ray.remote(PPOBuffer)
        return RemoteBuffer.remote(config)
    elif config.type == 'ppo':
        return PPOBuffer(config)
    elif config.type == 'local':
        return LocalBuffer(config)
    else:
        raise ValueError(config['training'])
