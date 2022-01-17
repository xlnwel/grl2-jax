import time
import logging
import collections
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from utility.utils import config_attr, dict2AttrDict, moments, standardize


logger = logging.getLogger(__name__)


def compute_nae(reward, discount, value, last_value, 
                gamma, mask=None, epsilon=1e-8):
    next_return = last_value
    traj_ret = np.zeros_like(reward)
    for i in reversed(range(reward.shape[0])):
        traj_ret[i] = next_return = (reward[i]
            + discount[i] * gamma * next_return)

    # Standardize traj_ret and advantages
    traj_ret_mean, traj_ret_var = moments(traj_ret)
    traj_ret_std = np.maximum(np.sqrt(traj_ret_var), 1e-8)
    value = standardize(value, mask=mask, epsilon=epsilon)
    # To have the same mean and std as trajectory return
    value = (value + traj_ret_mean) / traj_ret_std     
    advantage = standardize(traj_ret - value, mask=mask, epsilon=epsilon)
    traj_ret = standardize(traj_ret, mask=mask, epsilon=epsilon)

    return advantage, traj_ret

def compute_gae(reward, discount, value, last_value, gamma, 
                gae_discount, norm_adv=False, mask=None, epsilon=1e-8):
    if last_value is not None:
        last_value = np.expand_dims(last_value, 0)
        next_value = np.concatenate([value[1:], last_value], axis=0)
    else:
        next_value = value[1:]
        value = value[:-1]
    assert value.shape == next_value.shape, (value.shape, next_value.shape)
    advs = delta = (reward + discount * gamma * next_value - value).astype(np.float32)
    next_adv = 0
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] 
            + discount[i] * gae_discount * next_adv)
    traj_ret = advs + value
    if norm_adv:
        advs = standardize(advs, mask=mask, epsilon=epsilon)
    return advs, traj_ret

def compute_indices(idxes, mb_idx, mb_size, N_MBS):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % N_MBS
    curr_idxes = idxes[start: end]
    return mb_idx, curr_idxes

def get_sample(memory, idxes, sample_keys, state_keys, state_type=None):
    if state_type is None:
        sample = {k: memory[k][idxes, 0]
            if k in state_keys else memory[k][idxes] 
            for k in sample_keys}
    else:
        sample = {}
        state = []
        for k in sample_keys:
            if k in state_keys:
                v = memory[k][idxes, 0]
                state.append(v.reshape(-1, v.shape[-1]))
            else:
                sample[k] = memory[k][idxes]
        sample['state'] = state_type(*state)

    return sample


class LocalBuffer(Buffer):
    def __init__(
        self, 
        config: dict2AttrDict, 
        model: Model,
        n_players: int
    ):
        self.config = config

        self._state_keys = model.state_keys

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
                # if k in self._state_keys:
                #     assert v.shape[0] == self._n_envs * self._n_players, (k, v.shape, self._n_envs * self._n_players)
                # else:
                #     assert v.shape[:2] == (self._n_envs, self._n_players), (k, v.shape, (self._n_envs, self._n_players))
                self._buffers[i][k].append(v[i])
            self._buff_lens[i] += 1
        self._memlen += 1

    def retrieve_all_data(self, last_value):
        assert self._memlen == self._maxlen, (self._memlen, self._maxlen)
        episode = {k: np.stack([np.stack(b[k]) for b in self._buffers], 1)
            for k in self._buffers[0].keys()}
        # for k, v in episode.items():
        #     if k in self._state_keys:
        #         assert v.shape[:2] == (self._memlen, self._n_envs * self._n_players), \
        #             (k, v.shape, (self._memlen, self._n_envs * self._n_players))
        #     else:
        #         assert v.shape[:3] == (self._memlen, self._n_envs, self._n_players), \
        #             (k, v.shape, (self._memlen, self._n_envs, self._n_players))

        episode['advantage'], episode['traj_ret'] = compute_gae(
            episode['reward'], episode['discount'], episode['value'], 
            last_value, self.config.gamma, self._gae_discount)
        
        data = {k: np.swapaxes(episode[k], 0, 1) 
            for k in self.config.sample_keys}
        # for k, v in data.items():
        #     if k in self._state_keys:
        #         assert v.shape[:2] == (self._n_envs * self._n_players, self._memlen), \
        #             (k, v.shape, (self._n_envs * self._n_players, self._memlen))
        #     else:
        #         assert v.shape[:3] == (self._n_envs, self._memlen, self._n_players), \
        #             (k, v.shape, (self._n_envs, self._memlen, self._n_players))
        
        return data, self._memlen * self._n_envs

    def retrieve_episode(self, eid, last_value):
        last_value = np.zeros((self._n_players,), dtype=np.float32)
        episode = {k: np.stack(v) for k, v in self._buffers[eid].items()}
        # for k, v in episode.items():
        #     assert v.shape[:3] == (self._memlen, self._n_envs, self._n_players), \
        #         (k, v.shape, (self._memlen, self._n_envs, self._n_players))

        episode['advantage'], episode['traj_ret'] = compute_gae(
            episode['reward'], episode['discount'], episode['value'], 
            last_value, self.config.gamma, self._gae_discount)

        epslen = self._buff_lens[eid]
        self._buffers[eid] = collections.defaultdict(list)
        self._buff_lens[eid] = 0

        return episode, epslen


class PPOBuffer(Buffer):
    def __init__(
        self, 
        config: dict, 
        model: Model
    ):
        self.config = config_attr(self, config)
        self._add_attributes(model)

    def _add_attributes(self, model):
        self._norm_adv = self.config.get('norm_adv', 'minibatch')
        self._use_dataset = self.config.get('use_dataset', False)
        do_logging(f'Is dataset used for data pipline: {self._use_dataset}', logger=logger)

        self._sample_size = self.config.get('sample_size', None)
        self._sample_keys = self.config.sample_keys
        self._state_keys = model.state_keys
        self._state_type = model.state_type

        self._max_size = self.config.n_workers * self.config.n_envs * self.config.N_STEPS
        self._current_size = 0
        self._batch_size = self._max_size // self._sample_size if self._sample_size else self._max_size
        self._mb_size = self._batch_size // self.N_MBS
        self._idxes = np.arange(self._batch_size)
        self._shuffled_idxes = np.arange(self._batch_size)
        self._gamma = self.config.gamma
        self._gae_discount = self._gamma * self.config.lam
        self._epsilon = 1e-5
        self.reset()
        do_logging(f'Batch size: {self._batch_size}', logger=logger)
        do_logging(f'Mini-batch size: {self._mb_size}', logger=logger)

        self._sleep_time = 0.025
        self._sample_wait_time = 0
        self._epoch_idx = 0
        self._first_sample = True
        self.MAX_WAIT_TIME = self.config.get('MAX_WAIT_TIME', 60)

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
        self._epoch_idx = 0
        self._current_size = 0
        self._ready = False

    """ Filling Methods """
    def add(self, **transition):
        """ Add transitions """
        for k, v in transition.items():
            self._memory[k].append(v)
        self._current_size += 1

        if self._current_size == self.N_STEPS:
            for k, v in self._memory.items():
                self._memory[k] = np.stack(v)

    def finish(self, last_value):
        assert self._current_size == self.N_STEPS, self._current_size
        if self.config.adv_type == 'nae':
            assert self._norm_adv == 'batch', self._norm_adv
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_nae(
                reward=self._memory['reward'], 
                discount=self._memory['discount'],
                value=self._memory['value'],
                last_value=last_value,
                gamma=self._gamma,
                mask=self._memory.get('life_mask'),
                epsilon=self._epsilon)
        elif self.config.adv_type == 'gae':
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_gae(
                reward=self._memory['reward'], 
                discount=self._memory['discount'],
                value=self._memory['value'],
                last_value=last_value,
                gamma=self._gamma,
                gae_discount=self._gae_discount,
                norm_adv=self._norm_adv == 'batch',
                mask=self._memory.get('life_mask'),
                epsilon=self._epsilon)
        elif self.config.adv_type == 'vtrace':
            pass
        else:
            raise NotImplementedError

        for k, v in self._memory.items():
            v = np.swapaxes(v, 0, 1)
            self._memory[k] = np.reshape(v,
                (self._batch_size, self._sample_size, *v.shape[2:])
                if self._sample_size else (self._batch_size, *v.shape[2:]))
        self._ready = True

    def merge_data(self, data, n):
        """ Merging Data from Other Buffers """
        assert isinstance(self._memory, collections.defaultdict), self._memory
        for k, v in data.items():
            self._memory[k].append(v)
        self._current_size += n

        if self._current_size == self._max_size:
            for k, v in self._memory.items():
                v = np.concatenate(v)
                assert v.shape[:2] == (self.config.n_workers * self.config.n_envs, self.config.N_STEPS), \
                    (v.shape[:2], (self.config.n_workers * self.config.n_envs, self.config.N_STEPS))
                if self._sample_size:
                    self._memory[k] = np.reshape(v, 
                        (self._batch_size, self._sample_size, *v.shape[2:]))
                else:
                    self._memory[k] = np.reshape(v, 
                        (self._batch_size, *v.shape[2:]))
            self._ready = True

    """ Update Data """
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

        for start in range(0, self._batch_size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            if self._sample_size:
                state = tuple([self._memory[k][curr_idxes, 0] 
                    for k in self._state_keys])
                mask = self._memory['mask'][curr_idxes]
                value, state = fn(obs, state=state, mask=mask, return_state=True)
                self.update('value', value, mb_idxes=curr_idxes)
                next_idxes = curr_idxes + self._mb_size
                self.update('state', state, mb_idxes=next_idxes)
            else:
                value = fn(obs)
                self.update('value', value, mb_idxes=curr_idxes)
        
        assert mb_idx == 0, mb_idx

    """ Sampling """
    def sample(self, sample_keys=None):
        ready = self._wait_to_sample()
        if not ready:
            return None
        self._shuffle_indices()
        sample = self._sample(sample_keys)
        self._post_process_for_dataset()
        return sample

    """ Implementations """
    def _wait_to_sample(self):
        while not self._ready and (
                self._first_sample or self._sample_wait_time < self.MAX_WAIT_TIME):
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time
        # print(f'PPOBuffer starts sampling: waiting time: {self._sample_wait_time}', self._ready)
        self._sample_wait_time = 0
        return self._ready

    def _shuffle_indices(self):
        if self.N_MBS > 1 and self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        
    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, 
            self._mb_size, self.N_MBS)

        sample = get_sample(
            self._memory, self._curr_idxes, sample_keys,
            self._state_keys, self._state_type)
        sample = self._process_sample(sample)

        return sample

    def _process_sample(self, sample):
        if 'advantage' in sample and self._norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], mask=sample.get('life_mask'), 
                epsilon=self._epsilon)
        return sample
    
    def _post_process_for_dataset(self):
        if self._mb_idx == 0:
            self._epoch_idx += 1
            if self._epoch_idx == self.N_EPOCHS:
                # resetting here is especially important 
                # if we use tf.data as sampling is done 
                # in a background thread
                self.reset()

    def clear(self):
        self._memory = {}
        self.reset()

def create_buffer(config, model, **kwargs):
    config = dict2AttrDict(config)
    if config.type == 'ppo' or config.type == 'pbt':
        return PPOBuffer(config, model, **kwargs)
    elif config.type == 'local':
        return LocalBuffer(config, model, **kwargs)
    else:
        raise ValueError(config.type)
