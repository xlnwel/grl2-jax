import time
import logging
import collections
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from utility.utils import dict2AttrDict, moments, standardize


logger = logging.getLogger(__name__)


def compute_nae(
    reward, 
    discount, 
    value, 
    last_value, 
    gamma, 
    mask=None, 
    epsilon=1e-8
):
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

def compute_gae(
    reward, 
    discount, 
    value, 
    last_value, 
    gamma,
    gae_discount, 
    norm_adv=False, 
    mask=None, 
    epsilon=1e-8
):
    if last_value is not None:
        last_value = np.expand_dims(last_value, 0)
        next_value = np.concatenate([value[1:], last_value], axis=0)
    else:
        next_value = value[1:]
        value = value[:-1]
    next_value = np.mean(next_value, axis=2, keepdims=True)
    assert value.ndim == next_value.ndim, (value.shape, next_value.shape)
    advs = delta = (reward + discount * gamma * next_value - value).astype(np.float32)
    next_adv = 0
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] 
            + discount[i] * gae_discount * next_adv)
    traj_ret = advs + value
    if norm_adv:
        advs = standardize(advs, mask=mask, epsilon=epsilon)

    return advs, traj_ret

def compute_indices(idxes, mb_idx, mb_size, n_mbs):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % n_mbs
    curr_idxes = idxes[start: end]

    return mb_idx, curr_idxes

def get_sample(
    memory, 
    idxes, 
    sample_keys, 
    actor_state_keys, 
    value_state_keys, 
    actor_state_type,
    value_state_type,
):
    if actor_state_type is None and value_state_type is None:
        sample = {k: memory[k][idxes] for k in sample_keys}
    else:
        sample = {}
        actor_state = []
        value_state = []
        for k in sample_keys:
            if k in actor_state_keys:
                v = memory[k][idxes, 0]
                actor_state.append(v.reshape(-1, v.shape[-1]))
            elif k in value_state_keys:
                v = memory[k][idxes, 0]
                value_state.append(v.reshape(-1, v.shape[-1]))
            else:
                sample[k] = memory[k][idxes]
        if actor_state:
            sample['actor_state'] = actor_state_type(*actor_state)
        if value_state:
            sample['value_state'] = value_state_type(*value_state)

    return sample


def get_sample_keys_size(config):
    actor_state_keys = ['actor_h', 'actor_c']
    value_state_keys = ['value_h', 'value_c']
    if config['actor_rnn_type'] or config['value_rnn_type']: 
        sample_keys = config.sample_keys
        sample_size = config.sample_size
        if not config['actor_rnn_type']:
            sample_keys = _remote_state_keys(
                sample_keys, 
                actor_state_keys, 
            )
        if not config['value_rnn_type']:
            sample_keys = _remote_state_keys(
                sample_keys, 
                value_state_keys, 
            )
    else:
        sample_keys = _remote_state_keys(
            config.sample_keys, 
            actor_state_keys + value_state_keys, 
        )
        sample_keys.remove('mask')
        sample_size = None

    return sample_keys, sample_size


def _remote_state_keys(sample_keys, state_keys):
    for k in state_keys:
        if k in sample_keys:
            sample_keys.remove(k)

    return sample_keys


class LocalBuffer(Buffer):
    def __init__(
        self, 
        config: dict2AttrDict, 
        model: Model,
        runner_id: int,
        n_units: int,
    ):
        self.config = config
        self.runner_id = runner_id

        self.actor_state_keys = (f'actor_{k}' for k in model.actor_state_keys)
        self.value_state_keys = (f'value_{k}' for k in model.value_state_keys)
        self.sample_keys, self.sample_size = get_sample_keys_size(config)

        if self.config.get('fragment_size', None) is not None:
            self.maxlen = self.config.fragment_size
        elif self.sample_size is not None:
            self.maxlen = self.sample_size * 4
        else:
            self.maxlen = 64
        self.maxlen = min(self.maxlen, self.config.n_steps)
        if self.sample_size is not None:
            assert self.maxlen % self.sample_size == 0, (self.maxlen, self.sample_size)
            assert self.config.n_steps % self.maxlen == 0, (self.config.n_steps, self.maxlen)
            self.n_samples = self.maxlen // self.sample_size
        self.n_envs = self.config.n_envs
        self.gae_discount = self.config.gamma * self.config.lam
        self.n_units = n_units

        self.reset()

    def size(self):
        return self._memlen

    def is_empty(self):
        return self._memlen == 0

    def is_full(self):
        return self._memlen == self.maxlen

    def reset(self):
        self._memlen = 0
        self._buffers = [collections.defaultdict(list) for _ in range(self.n_envs)]
        # self._train_steps = [[None for _ in range(self.n_units)] 
        #     for _ in range(self.n_envs)]

    def add(self, data):
        for k, v in data.items():
            assert v.shape[:2] == (self.n_envs, self.n_units), \
                (k, v.shape, (self.n_envs, self.n_units))
            for i in range(self.n_envs):
                self._buffers[i][k].append(v[i])
        self._memlen += 1

    def retrieve_all_data(self, last_value):
        assert self._memlen == self.maxlen, (self._memlen, self.maxlen)
        data = {k: np.stack([np.stack(b[k]) for b in self._buffers], 1)
            for k in self._buffers[0].keys()}
        for k, v in data.items():
            assert v.shape[:3] == (self._memlen, self.n_envs, self.n_units), \
                (k, v.shape, (self._memlen, self.n_envs, self.n_units))

        data['advantage'], data['traj_ret'] = compute_gae(
            reward=data['reward'], 
            discount=data['discount'], 
            value=data['value'], 
            last_value=last_value, 
            gamma=self.config.gamma, 
            gae_discount=self.gae_discount
        )

        data = {k: np.swapaxes(data[k], 0, 1) for k in self.sample_keys}
        for k, v in data.items():
            assert v.shape[:3] == (self.n_envs, self._memlen, self.n_units), \
                (k, v.shape, (self.n_envs, self._memlen, self.n_units))
            if self.sample_size is not None:
                data[k] = np.reshape(v, 
                    (self.n_envs * self.n_samples, self.sample_size, *v.shape[2:]))
            else:
                data[k] = np.reshape(v,
                    (self.n_envs * self._memlen, *v.shape[2:]))
        
        self.reset()
        
        return self.runner_id, data, self.n_envs * self.maxlen


class PPOBuffer(Buffer):
    def __init__(
        self, 
        config: dict, 
        model: Model
    ):
        self.config = dict2AttrDict(config)
        self._add_attributes(model)

    def _add_attributes(self, model):
        self.norm_adv = self.config.get('norm_adv', 'minibatch')
        self.use_dataset = self.config.get('use_dataset', False)
        do_logging(f'Is dataset used for data pipline: {self.use_dataset}', logger=logger)

        self.actor_state_keys = tuple([f'actor_{k}' for k in model.actor_state_keys])
        self.value_state_keys = tuple([f'value_{k}' for k in model.value_state_keys])
        self.actor_state_type = model.actor_state_type
        self.value_state_type = model.value_state_type
        self.sample_keys, self.sample_size = get_sample_keys_size(self.config)

        self.n_runners = self.config.n_runners
        self.n_steps = self.config.n_steps
        self.max_size = self.config.get('max_size', self.n_runners * self.config.n_envs * self.n_steps)
        if self.sample_size:
            assert self.max_size % self.sample_size == 0, (self.max_size, self.sample_size)
        self.batch_size = self.max_size // self.sample_size if self.sample_size else self.max_size
        self.n_epochs = self.config.n_epochs
        self.n_mbs = self.config.n_mbs
        self.mb_size = self.batch_size // self.n_mbs
        self.idxes = np.arange(self.batch_size)
        self.shuffled_idxes = np.arange(self.batch_size)
        self.gamma = self.config.gamma
        self.gae_discount = self.gamma * self.config.lam
        self.epsilon = 1e-5
        do_logging(f'Batch size: {self.batch_size}', logger=logger)
        do_logging(f'Mini-batch size: {self.mb_size}', logger=logger)

        self.reset()

        self._sleep_time = 0.025
        self._sample_wait_time = 0
        self.max_wait_time = self.config.get('max_wait_time', 5)

    def __getitem__(self, k):
        return self._memory[k]

    def __contains__(self, k):
        return k in self._memory
    
    def size(self):
        return self._current_size

    def ready(self):
        return self._ready

    def reset(self):
        self._buffers = [collections.defaultdict(list) for _ in range(self.n_runners)]
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

        if self._current_size == self.n_steps:
            for k, v in self._memory.items():
                self._memory[k] = np.stack(v)

    def finish(self, last_value):
        assert self._current_size == self.n_steps, self._current_size
        if self.config.adv_type == 'nae':
            assert self.norm_adv == 'batch', self.norm_adv
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_nae(
                reward=self._memory['reward'], 
                discount=self._memory['discount'],
                value=self._memory['value'],
                last_value=last_value,
                gamma=self.gamma,
                mask=self._memory.get('life_mask'),
                epsilon=self.epsilon)
        elif self.config.adv_type == 'gae':
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_gae(
                reward=self._memory['reward'], 
                discount=self._memory['discount'],
                value=self._memory['value'],
                last_value=last_value,
                gamma=self.gamma,
                gae_discount=self.gae_discount,
                norm_adv=self.norm_adv == 'batch',
                mask=self._memory.get('life_mask'),
                epsilon=self.epsilon)
        elif self.config.adv_type == 'vtrace':
            pass
        else:
            raise NotImplementedError

        for k, v in self._memory.items():
            v = np.swapaxes(v, 0, 1)
            self._memory[k] = np.reshape(v,
                (self.batch_size, self.sample_size, *v.shape[2:])
                if self.sample_size else (self.batch_size, *v.shape[2:]))
        self._ready = True

    def merge_data(self, rid: int, data: dict, n: int):
        """ Merging Data from Other Buffers """
        if self._memory:
            # neglects superfluous data
            return
        assert self._memory == {}, list(self._memory)
        assert not self._ready, (self._current_size, self.max_size, n)
        for k, v in data.items():
            self._buffers[rid][k].append(v)
        self._current_size += n

        if self._current_size >= self.max_size:
            self._memory = {k: np.concatenate(
                    [np.concatenate(b[k]) for b in self._buffers if b])[-self.batch_size:]
                for k in self.sample_keys}
            for k, v in self._memory.items():
                assert v.shape[0] == self.batch_size, (v.shape, self.batch_size)
            self._ready = True
            self._buffers = [collections.defaultdict(list) for _ in range(self.n_runners)]

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

        for start in range(0, self.batch_size, self.mb_size):
            end = start + self.mb_size
            curr_idxes = self.idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            if self.sample_size:
                state = tuple([self._memory[k][curr_idxes, 0] 
                    for k in self.state_keys])
                mask = self._memory['mask'][curr_idxes]
                value, state = fn(obs, state=state, mask=mask, return_state=True)
                self.update('value', value, mb_idxes=curr_idxes)
                next_idxes = curr_idxes + self.mb_size
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
                self._sample_wait_time < self.max_wait_time):
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time
        # print(f'PPOBuffer starts sampling: waiting time: {self._sample_wait_time}', self._ready)
        # if not self._ready:
        #     raise RuntimeError(f'No data received in time {self.max_wait_time}; Elapsed time: {self._sample_wait_time}')
        self._sample_wait_time = 0
        return self._ready

    def _shuffle_indices(self):
        if self.n_mbs > 1 and self._mb_idx == 0:
            np.random.shuffle(self.shuffled_idxes)

    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self.sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self.shuffled_idxes, self._mb_idx, 
            self.mb_size, self.n_mbs)

        sample = get_sample(
            self._memory, 
            self._curr_idxes, 
            sample_keys,
            self.actor_state_keys, 
            self.value_state_keys,
            self.actor_state_type, 
            self.value_state_type
        )
        sample = self._process_sample(sample)

        return sample

    def _process_sample(self, sample):
        if 'advantage' in sample and self.norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], mask=sample.get('life_mask'), 
                epsilon=self.epsilon)
        return sample

    def _post_process_for_dataset(self):
        if self._mb_idx == 0:
            self._epoch_idx += 1
            if self._epoch_idx == self.n_epochs:
                # resetting here is especially important 
                # if we use tf.data as sampling is done 
                # in a background thread
                self.reset()

    def clear(self):
        self.reset()

def create_buffer(config, model, **kwargs):
    config = dict2AttrDict(config)
    if config.type == 'ppo':
        return PPOBuffer(config, model, **kwargs)
    elif config.type == 'local':
        return LocalBuffer(config, model, **kwargs)
    else:
        raise ValueError(config.type)
