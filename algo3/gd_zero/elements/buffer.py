import collections
import time
import logging
import numpy as np

from core.log import do_logging
from utility.utils import batch_dicts, config_attr, dict2AttrDict, standardize


logger = logging.getLogger(__name__)


def compute_gae(reward, discount, value, gamma, gae_discount):
    next_value = np.concatenate([value[1:], [0]], axis=0).astype(np.float32)
    assert value.shape == next_value.shape, (value.shape, next_value.shape)
    advs = delta = (reward + discount * gamma * next_value - value).astype(np.float32)
    # advs2 = (reward + discount * next_value - value)
    next_adv = 0
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] 
            + discount[i] * gae_discount * next_adv)
        # advs2[i] = next_adv2 = reward[i] + discount[i] * 
    traj_ret = advs + value

    return advs, traj_ret


def compute_indices(idxes, mb_idx, mb_size, n_mbs):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % n_mbs
    curr_idxes = idxes[start: end]
    return mb_idx, curr_idxes


def get_sample(memory, sample_keys, state_keys, idxes):
    sample = {k: memory[k][idxes, 0]
        if k in state_keys else memory[k][idxes] 
        for k in sample_keys}
    action_rnn_dim = sample['action_h'].shape[-1]
    sample['action_h'] = sample['action_h'].reshape(-1, action_rnn_dim)
    sample['action_c'] = sample['action_c'].reshape(-1, action_rnn_dim)
    return sample


class LocalBuffer:
    def __init__(self, config):
        self.config = dict2AttrDict(config)
        self._gae_discount = self.config.gamma * self.config.lam
        self._maxlen = self.config.n_envs * self.config.n_steps
        self.reset()

    def size(self):
        return self._memlen

    def reset(self):
        self._memory = []
        self._memlen = 0
        self._buffer = collections.defaultdict(lambda: collections.defaultdict(list))
        self._buff_lens = collections.defaultdict(int)

    def add(self, data):
        eids = data.pop('eid')
        pids = data.pop('pid')
        reward = data['reward']
        discount = data['discount']
        mask = data['mask']
        for i, (eid, pid, r, d, m) in enumerate(zip(eids, pids, reward, discount, mask)):
            if m == 0:
                assert self._buff_lens[(eid, pid)] == 0, (eid, pid, self._buff_lens[(eid, pid)])
                assert len(self._buffer[(eid, pid)]) == 0, (eid, pid, len(self._buffer[(eid, pid)])) 
            assert len(r) == 4, r
            for k, vs in data.items():
                if k == 'reward':
                    self._buffer[(eid, pid)][k].append(vs[i][pid])
                else:
                    self._buffer[(eid, pid)][k].append(vs[i])
            self._buff_lens[(eid, pid)] += 1
            assert np.all(r == 0) or d == 0, (r, d)

    def finish(self, eids, rewards):
        assert len(eids) == len(rewards), (eids, rewards)
        for eid, reward in zip(eids, rewards):
            for pid in self.agent_pids:
                self._buffer[(eid, pid)]['reward'][-1] = reward[pid]
                self._buffer[(eid, pid)]['discount'][-1] = 0
                assert self._buffer[(eid, pid)]['reward'][-1] == reward[pid], self._buffer[(eid, pid)]['reward'][-1]
                assert self._buffer[(eid, pid)]['discount'][-1] == 0, self._buffer[(eid, pid)]['discount'][-1]
                self.merge_episode(eid, pid)

    def merge_episode(self, eid, pid):
        episode = {k: np.stack(v) for k, v in self._buffer[(eid, pid)].items()}
        epslen = self._buff_lens[(eid, pid)]
        for k, v in episode.items():
            assert v.shape[0] == epslen, (v.shape, epslen)
            if k == 'discount':
                assert np.all(v[:-1] == 1), v
                assert np.all(v[-1] == 0), v
            elif k == 'mask':
                assert np.all(v[1:] == 1), v
                assert np.all(v[0] == 0), v
        episode['advantage'], episode['traj_ret'] = compute_gae(
            episode['reward'], episode['discount'], episode['value'], 
            self.config.gamma, self._gae_discount
        )
        episode = {k: v for k, v in episode.items()}
        self._memory.append(episode)
        self._memlen += epslen
        self._buffer[(eid, pid)] = collections.defaultdict(list)
        self._buff_lens[(eid, pid)] = 0

    def ready_to_retrieve(self):
        return self._memlen > self._maxlen

    def retrieve_data(self):
        data = batch_dicts(self._memory, np.concatenate)
        for v in data.values():
            assert v.shape[0] == self._memlen, (v.shape, self._memlen)
        self.reset()
        return data

    def set_pids(self, agent_pids, other_pids):
        self.agent_pids = agent_pids
        self.other_pids = other_pids


class PPOBuffer:
    def __init__(self, config):
        self.config = config_attr(self, config)
        self._use_dataset = getattr(self, '_use_dataset', False)
        if self._use_dataset:
            do_logging(f'Dataset is used for data pipline', logger=logger)

        self._sample_size = getattr(self, '_sample_size', None)
        self._state_keys = ['h', 'c']
        assert self._n_envs * self.n_steps % self._sample_size == 0, \
            f'{self._n_envs} * {self.n_steps} % {self._sample_size} != 0'
        do_logging(f'Sample size: {self._sample_size}', logger=logger)

        self._max_size = self._n_envs * self.n_steps
        self._batch_size = self._max_size // self._sample_size
        self._mb_size = self._batch_size // self.n_mbs
        self._idxes = np.arange(self._batch_size)
        self._shuffled_idxes = np.arange(self._batch_size)
        self._gae_discount = self._gamma * self._lam
        self._epsilon = 1e-5
        if hasattr(self, 'N_VALUE_EPOCHS'):
            self.n_epochs += self.N_VALUE_EPOCHS
        self.reset()
        do_logging(f'Batch size: {self._batch_size}', logger=logger)
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
    
    def type(self):
        return 'ppo'

    def ready(self):
        return self._ready

    def max_size(self):
        return self._max_size

    def reset(self):
        self._memory = collections.defaultdict(list)
        self._mb_idx = 0
        self._epoch_idx = 0
        self._ready = False

    def merge(self, data):
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
            v = np.concatenate(v)
            v = v[:self._batch_size * self._sample_size]
            self._memory[k] = v.reshape(self._batch_size, self._sample_size, *v.shape[1:])

        self._ready = True
        return self._memory

    """ Implementations """
    def _wait_to_sample(self):
        while not self._ready:
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time

    def _shuffle_indices(self):
        if self.n_mbs > 1 and self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        
    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, 
            self._mb_size, self.n_mbs)

        sample = get_sample(self._memory, sample_keys, self._state_keys, self._curr_idxes)
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
            if self._epoch_idx == self.n_epochs:
                # resetting here is especially important 
                # if we use tf.data as sampling is done 
                # in a background thread
                self.reset()


class PPGBuffer:
    def __init__(self, config):
        self.config = config_attr(self, config)
        config['use_dataset'] = False
        self._buff = PPOBuffer(config)

        self._sample_size = self.config.sample_size
        self._sample_keys = self.config.sample_keys
        self._aux_compute_keys = self.config.aux_compute_keys
        self._state_keys = ['h', 'c']

        assert self.n_pi >= self.n_segs, (self.n_pi, self.n_segs)
        buff_size = self._buff.max_size()
        self._size = buff_size * self.n_segs
        self._batch_size = self._size // self._sample_size
        assert self._size == self._batch_size * self._sample_size, \
            (self._size, self._batch_size, self._sample_size)
        self._aux_mb_size = self._batch_size // self.n_segs // self.n_aux_mbs_per_seg
        assert self._batch_size == self.n_aux_mbs_per_seg * self.n_segs * self._aux_mb_size, \
            (self._batch_size, self._aux_mb_size, self.n_aux_mbs_per_seg, self.n_segs)
        self.N_AUX_MBS = self.n_segs * self.n_aux_mbs_per_seg
        self._shuffled_idxes = np.arange(self._batch_size)
        self._idx = 0
        self._mb_idx = 0
        self._epoch_idx = 0
        self._ready = False
        self._ready_for_aux_training = False

        self._gae_discount = self._gamma * self._lam
        self._memory = collections.defaultdict(list)
        do_logging(f'Memory size: {self._size}', logger=logger)
        do_logging(f'Batch size: {self._size}', logger=logger)
        do_logging(f'Aux mini-batch size: {self._aux_mb_size}', logger=logger)

        self._sleep_time = 0.025
        self._sample_wait_time = 0

    def __getitem__(self, k):
        return self._buff[k]

    def ready(self):
        return self._ready
    
    def ready_for_aux_training(self):
        return self._ready_for_aux_training

    def type(self):
        return 'ppg'

    def reset(self):
        self._buff.reset()
        assert self._ready, self._idx
        self._memory.clear()
        self._idx = 0
        self._mb_idx = 0
        self._epoch_idx = 0
        self._ready = False
        self._ready_for_aux_training = False

    def merge(self, data):
        self._buff.merge(data)

    def sample(self):
        def wait_to_sample():
            while not (self._ready or self._buff.ready()):
                time.sleep(self._sleep_time)
                self._sample_wait_time += self._sleep_time

        def shuffle_indices():
            if self._mb_idx == 0:
                np.random.shuffle(self._shuffled_idxes)

        def sample_minibatch(keys):
            self._mb_idx, idxes = compute_indices(
                self._shuffled_idxes, self._mb_idx, 
                self._aux_mb_size, self.N_AUX_MBS)
            
            sample = get_sample(self._memory, keys, self._state_keys, idxes)

            return sample

        def post_process_for_dataset():
            if self._mb_idx == 0:
                self._epoch_idx += 1
                if self._epoch_idx == self.n_aux_epochs:
                    # resetting here is especially important 
                    # if we use tf.data as sampling is done 
                    # in a background thread
                    self.reset()

        wait_to_sample()
        if self._buff.ready():
            sample = self._buff.sample()
        else:
            assert self._ready and self._idx == 0, (self._ready, self._idx)
            shuffle_indices()
            sample = sample_minibatch(self._aux_sample_keys)
            post_process_for_dataset()

        return sample

    def compute_mean_max_std(self, stats='reward'):
        return self._buff.compute_mean_max_std(stats)
    
    def finish(self):
        def transfer_data(data):
            assert self._buff.ready(), (self._buff.size(), self._buff.max_size())
            if self._idx >= self.n_pi - self.n_segs:
                for k, v in data.items():
                    self._memory[k].append(v)
            self._idx = (self._idx + 1) % self.n_pi

        def aux_finish():
            for k, v in self._memory.items():
                assert len(v) == self.n_segs, (len(v), self.n_segs)
                v = np.concatenate(v)
                assert v.shape[0] == self._batch_size, (v.shape, self._batch_size)
                self._memory[k] = v[-self._batch_size:]

        data = self._buff.finish()
        transfer_data(data)
        if self._idx == 0:
            self._ready_for_aux_training = True
            aux_finish()

    def compute_aux_data_with_func(self, func):
        assert self._idx == 0, self._idx
        action_type_logits_list = []
        card_rank_logits_list = []
        value_list = []
        start = 0

        for _ in range(self.n_segs * self.n_aux_mbs_per_seg):
            end = start + self._aux_mb_size
            sample = get_sample(
                self._memory, self._aux_compute_keys, self._state_keys, np.arange(start, end))
            action_type_logits, card_rank_logits, value = func(sample)
            action_type_logits_list.append(action_type_logits)
            card_rank_logits_list.append(card_rank_logits)
            value_list.append(value)
            start = end
        assert start == self._batch_size, (start, self._batch_size)

        self._memory['value'] = np.concatenate(value_list)
        self._memory['action_type_logits'] = np.concatenate(action_type_logits_list)
        self._memory['card_rank_logits'] = np.concatenate(card_rank_logits_list)

        reward = self._memory['reward'].reshape(-1)
        discount = self._memory['discount'].reshape(-1)
        value = self._memory['value'].reshape(-1)
        _, traj_ret = compute_gae(reward, discount, value, self._gamma, self._gae_discount)
        self._memory['traj_ret'] = traj_ret.reshape(self._batch_size, self._sample_size)
        self._ready = True
        self._ready_for_aux_training = False


def create_buffer(config, central_buffer=False):
    config = dict2AttrDict(config)
    if central_buffer:
        assert config.type == 'ppo', config.type
        import ray
        RemoteBuffer = ray.remote(PPOBuffer)
        return RemoteBuffer.remote(config)
    elif config.type =='ppg' or config.type == 'pbt':
        return PPGBuffer(config)
    elif config.type == 'ppo':
        return PPOBuffer(config)
    elif config.type == 'local':
        return LocalBuffer(config)
    else:
        raise ValueError(config.type)


if __name__ == '__main__':
    n_steps = 3
    n_epochs = 2
    n_mbs = 2
    config = dict(
        type='ppg',
        gamma=1,
        lam=1,
        n_workers=2,
        n_envs=2,
        sample_size=2,
        n_steps=n_steps,
        n_epochs=n_epochs,
        N_VALUE_EPOCHS=0,
        n_mbs=n_mbs,
        N_AUX_MBS=n_mbs,
        n_pi=2,
        n_segs=2,
        sample_keys=['reward', 'value', 'action_h', 'action_c'],
        norm_adv='batch',
    )

    buffer = create_buffer(config)
    
    for _ in range(2):
        for k in range(n_steps):
            buffer.merge({
                'reward': np.array([k+i // 10 for i in range(3)]),
                'value': np.array([k+i // 10 for i in range(3)]),
                'action_h': np.zeros((3, 2)),
                'action_c': np.zeros((3, 2)),
            })
        buffer.finish()
        for _ in range(n_epochs * n_mbs):
            buffer.sample()
    
    print(buffer._ready_for_aux_training)
