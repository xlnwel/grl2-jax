import time
import logging
import collections
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from utility.typing import AttrDict
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
    epsilon=1e-8, 
    same_next_value=False
):
    if last_value is not None:
        last_value = np.expand_dims(last_value, 0)
        next_value = np.concatenate([value[1:], last_value], axis=0)
    else:
        next_value = value[1:]
        value = value[:-1]
    if same_next_value:
        next_value = np.mean(next_value, axis=-1, keepdims=True)
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

def compute_upgo(
    reward, 
    discount, 
    value, 
    last_value, 
    gamma,
    gae_discount, 
    norm_adv=False, 
    mask=None, 
    epsilon=1e-8, 
    same_next_value=False
):
    if last_value is not None:
        last_value = np.expand_dims(last_value, 0)
        next_value = np.concatenate([value[1:], last_value], axis=0)
    else:
        next_value = value[1:]
        value = value[:-1]
    if same_next_value:
        next_value = np.mean(next_value, axis=-1, keepdims=True)
    assert value.ndim == next_value.ndim, (value.shape, next_value.shape)
    rets = reward + discount * gamma * next_value
    next_ret = rets[-1]
    for i in reversed(range(advs.shape[0]))[1:]:
        rets[i] = next_ret = np.where(
            reward[i] + discount * gamma * next_ret > value[i], 
            reward[i] + discount * gamma * next_ret, rets[i]
        )
    advs = rets - value
    if norm_adv:
        advs = standardize(advs, mask=mask, epsilon=epsilon)

    return advs

def compute_bounded_target(prob, adv, config, z=0):
    c = config.get('c', .2)
    lower_clip = config.get('lower_clip', c)
    upper_clip = config.get('upper_clip', c)
    target_prob = prob * np.exp(adv / config.tau - z)
    tr_prob = np.clip(
        target_prob, 
        target_prob if lower_clip is None else prob-lower_clip, 
        target_prob if upper_clip is None else prob+upper_clip
    )
    if config.valid_clip:
        tr_prob = np.clip(tr_prob, 0, 1)
    return target_prob, tr_prob

def compute_clipped_target(prob, adv, config, z=0):
    c = config.get('c', .2)
    lower_clip = config.get('lower_clip', c)
    upper_clip = config.get('upper_clip', c)
    exp_adv = np.exp(adv / config.tau - z)
    target_prob = prob * exp_adv
    tr_prob = prob * np.clip(
        exp_adv, 
        exp_adv if upper_clip is None else 1-lower_clip, 
        exp_adv if upper_clip is None else 1+upper_clip
    )
    if config.valid_clip:
        tr_prob = np.clip(tr_prob, 0, 1)
    return target_prob, tr_prob

def compute_mixture_target(prob, adv, config, z=0):
    target_prob = prob * np.exp(adv / config.tau - z)
    tr_prob = (1 - config.alpha) * prob + config.alpha * target_prob
    if config.valid_clip:
        tr_prob = np.clip(tr_prob, 0, 1)
    return target_prob, tr_prob

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
        if 'mask' in sample_keys:
            sample_keys.remove('mask')
        sample_size = None

    return sample_keys, sample_size


def _remote_state_keys(sample_keys, state_keys):
    for k in state_keys:
        if k in sample_keys:
            sample_keys.remove(k)

    return sample_keys


class AdvantageCalculator:
    def compute_adv(self, config, last_value, data):
        if config.adv_type == 'gae':
            data['advantage'], data['traj_ret'] = \
                compute_gae(
                reward=data['reward'], 
                discount=data['discount'],
                value=data['value'],
                last_value=last_value,
                gamma=config.gamma,
                gae_discount=config.gae_discount,
                norm_adv=config.norm_adv == 'batch',
                mask=data.get('life_mask'),
                epsilon=config.epsilon,
                same_next_value=config.get('same_next_value', False)
            )
            data['raw_adv'] = data['advantage']
        elif config.adv_type == 'vtrace':
            pass
        else:
            raise NotImplementedError


class TargetProbabilityCalculator:
    def compute_target_tr_probs(self, config, data):
        compute_func = {
            'bounded': compute_bounded_target, 
            'clip': compute_clipped_target, 
            'mix': compute_mixture_target, 
        }[config.target_type]
        prob = np.exp(data['logprob'])
        data['target_prob'], data['tr_prob'] = compute_func(
            prob, data['advantage'], config
        )
        if 'target_pi' in config['sample_keys']:
            target_pi = data['pi'].copy()
            if data['action'].ndim == 1:
                idx = (
                    np.arange(data['action'].shape[0]),
                    data['action']
                )
            elif data['action'].ndim == 2:
                d1, d2 = data['action'].shape
                idx = (
                    np.tile(np.arange(d1)[:, None], [1, d2]), 
                    np.tile(np.arange(d2)[None, :], [d1, 1]), 
                    data['action']
                )
            elif data['action'].ndim == 3:
                d1, d2, d3 = data['action'].shape
                idx = (
                    np.tile(np.arange(d1)[:, None, None], [1, d2, d3]), 
                    np.tile(np.arange(d2)[None, :, None], [d1, 1, d3]), 
                    np.tile(np.arange(d3)[None, None, :], [d1, d2, 1]), 
                    data['action']
                )
            else:
                raise NotImplementedError(f'No support for dimensionality higher than 3, got {data["action"].shape}')
            target_pi[idx] = data['tr_prob']
            target_pi = target_pi / np.sum(target_pi, axis=-1, keepdims=True)
            data['target_pi'] = target_pi


class LocalBufferBase(TargetProbabilityCalculator, Buffer):
    def __init__(
        self, 
        config: AttrDict,
        env_stats: AttrDict,  
        model: Model,
        runner_id: int,
        n_units: int,
    ):
        self.config = config
        self.config.gae_discount = self.config.gamma * self.config.lam
        self.config.epsilon = self.config.get('epsilon', 1e-5)
        self.config.valid_clip = self.config.get(
            'valid_clip', env_stats.is_action_discrete)
        self.runner_id = runner_id
        self.n_units = n_units

        self._add_attributes(model)

    def _add_attributes(self, model):
        self.actor_state_keys = (f'actor_{k}' for k in model.actor_state_keys)
        self.value_state_keys = (f'value_{k}' for k in model.value_state_keys)
        self.sample_keys, self.sample_size = get_sample_keys_size(self.config)

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

    def add(self, data: dict):
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

        self.compute_adv(self.config, last_value, data)
        self.compute_target_tr_probs(self.config, data)

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


class PPOBufferBase(TargetProbabilityCalculator, Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
    ):
        self.config = dict2AttrDict(config)
        self.config.gae_discount = self.config.gamma * self.config.lam
        self.config.epsilon = self.config.get('epsilon', 1e-5)
        self.config.is_action_discrete = env_stats.is_action_discrete

        self._add_attributes(model)

    def _add_attributes(self, model):
        self.norm_adv = self.config.get('norm_adv', 'minibatch')
        self.use_dataset = self.config.get('use_dataset', False)
        do_logging(f'Is dataset used for data pipeline: {self.use_dataset}', logger=logger)

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
    def add(self, **data):
        """ Add transitions """
        for k, v in data.items():
            self._memory[k].append(v)
        self._current_size += 1

        if self._current_size == self.n_steps:
            for k, v in self._memory.items():
                self._memory[k] = np.stack(v)

    def finish(self, last_value):
        assert self._current_size == self.n_steps, self._current_size
        self.compute_adv(self.config, last_value, self._memory)
        self.compute_target_tr_probs(self.config, self._memory)

        for k, v in self._memory.items():
            assert v.shape[:2] == (self.config.n_steps, self.config.n_envs), (k, v.shape, (self.config.n_steps, self.config.n_envs))
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
            self.shuffled_idxes, 
            self._mb_idx, 
            self.mb_size, 
            self.n_mbs
        )

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
                epsilon=self.config.get('epsilon', 1e-5))
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

class LocalBuffer(AdvantageCalculator, LocalBufferBase):
    pass


class PPOBuffer(AdvantageCalculator, PPOBufferBase):
    pass


def create_buffer(config, model, env_stats, **kwargs):
    config = dict2AttrDict(config)
    env_stats = dict2AttrDict(env_stats)
    BufferCls = {
        'ppo': PPOBuffer, 
        'local': LocalBuffer
    }[config.type]
    return BufferCls(
        config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )
