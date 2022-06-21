import time
import logging
import collections
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from utility import div
from utility.rms import RunningMeanStd
from utility.typing import AttrDict
from utility.utils import batch_dicts, dict2AttrDict, moments, standardize


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
    gamma,
    gae_discount, 
    last_value=None, 
    next_value=None, 
    reset=None, 
    norm_adv=False, 
    mask=None, 
    epsilon=1e-8
):
    if reset is not None:
        assert next_value is not None, f'next_value is required when reset is given'
    if next_value is not None:
        pass
    elif last_value is not None:
        last_value = np.expand_dims(last_value, 0)
        next_value = np.concatenate([value[1:], last_value], axis=0)
    else:
        next_value = value[1:]
        value = value[:-1]
    assert reward.shape == discount.shape == value.shape == next_value.shape, (reward.shape, discount.shape, value.shape, next_value.shape)
    delta = (reward + discount * gamma * next_value - value).astype(np.float32)
    discount = (discount if reset is None else (1 - reset)) * gae_discount
    next_adv = 0
    advs = np.zeros_like(reward, dtype=np.float32)
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] + discount[i] * next_adv)
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

def normalize_adv(
    adv, 
    data, 
    normalize_adv=False, 
    zero_center=True, 
    process_adv=None, 
    adv_clip=None, 
):
    if normalize_adv:
        adv = standardize(
            adv, 
            zero_center, 
            data.get('life_mask'), 
        )
    if process_adv == 'tanh':
        adv_max = np.max(adv)
        adv = adv_max * np.tanh(2 * adv / adv_max)
    elif process_adv is None:
        pass
    else:
        raise NotImplementedError(f'Unknown {process_adv}')
    if adv_clip is not None:
        adv = np.clip(adv, -adv_clip, adv_clip)
    return adv

def compute_target_pi(data, config):
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

    target_prob_key = config.get('target_prob_key', 'target_prob_prime')
    target_pi[idx] = data[target_prob_key]
    target_pi = target_pi / np.sum(target_pi, axis=-1, keepdims=True)
    data['target_pi'] = target_pi
    data['vt_prob'] = target_pi[idx]

def add_probs_to_data(data, target_prob_prime, tr_prob_prime):
    data['target_prob_prime'] = target_prob_prime
    data['tr_prob_prime'] = tr_prob_prime
    data['target_prob'] = np.clip(target_prob_prime, 0, 1)
    data['tr_prob'] = np.clip(tr_prob_prime, 0, 1)

def compute_pr_clipped_target(
    data, 
    config, 
    z=0,
):
    adv = data['advantage']
    prob = np.exp(data['logprob'])
    lower_clip = config['pr_lower_clip']
    upper_clip = config['pr_upper_clip']

    target_prob_prime = prob * np.exp(adv / config.adv_tau - z)
    tr_prob_prime = np.clip(
        target_prob_prime, 
        target_prob_prime if lower_clip is None else prob-lower_clip, 
        target_prob_prime if upper_clip is None else prob+upper_clip
    )

    add_probs_to_data(data, target_prob_prime, tr_prob_prime)

def compute_exp_clipped_target(
    data, 
    config, 
    z=0
):
    adv = data['advantage']
    prob = np.exp(data['logprob'])
    lower_clip = config['exp_lower_clip']
    upper_clip = config['exp_upper_clip']

    exp_adv = np.exp(adv / config.adv_tau - z)
    target_prob_prime = prob * exp_adv
    tr_prob_prime = prob * np.clip(
        exp_adv, 
        exp_adv if lower_clip is None else 1-lower_clip, 
        exp_adv if upper_clip is None else 1+upper_clip
    )

    add_probs_to_data(data, target_prob_prime, tr_prob_prime)

def compute_tsallis_clipped_target(
    data, 
    config, 
    z=0
):
    adv = data['advantage']
    prob = np.exp(data['logprob'])
    lower_clip = config['tsallis_lower_clip']
    upper_clip = config['tsallis_upper_clip']

    exp_adv = div.tsallis_exp(adv / config.adv_tau - z)
    target_prob_prime = prob * exp_adv
    tr_prob_prime = prob * np.clip(
        exp_adv, 
        exp_adv if lower_clip is None else 1-lower_clip, 
        exp_adv if upper_clip is None else 1+upper_clip
    )

    add_probs_to_data(data, target_prob_prime, tr_prob_prime)

def compute_adv_clipped_target(
    data, 
    config, 
    z=0
):
    adv = data['advantage']
    prob = np.exp(data['logprob'])
    lower_clip = config['adv_lower_clip']
    upper_clip = config['adv_upper_clip']

    target_prob_prime = prob * np.exp(adv / config.adv_tau - z)
    tr_prob_prime = prob * np.exp(np.clip(
        adv, 
        adv if lower_clip is None else -lower_clip, 
        adv if upper_clip is None else upper_clip
    ) / config.adv_tau - z)

    add_probs_to_data(data, target_prob_prime, tr_prob_prime)

def compute_linear_clipped_target(
    data, 
    config, 
    z=0
):
    adv = data['advantage']
    prob = np.exp(data['logprob'])
    lower_clip = config['lin_lower_clip']
    upper_clip = config['lin_upper_clip']

    g = config['lin_weights'] * adv / np.maximum(prob, .1)
    target_prob_prime = g + prob
    tr_prob_prime = prob + np.clip(
        g, 
        g if lower_clip is None else -lower_clip, 
        g if upper_clip is None else upper_clip
    )

    add_probs_to_data(data, target_prob_prime, tr_prob_prime)

def compute_mixture_target(
    data, 
    config, 
    z=0
):
    adv = data['advantage']
    prob = np.exp(data['logprob'])

    target_prob_prime = prob * np.exp(adv / config.adv_tau - z)
    tr_prob_prime = (1 - config.alpha) * prob + config.alpha * target_prob_prime

    add_probs_to_data(data, target_prob_prime, tr_prob_prime)

def compute_step_target(
    data, 
    config, 
    z=0
):
    adv = data['advantage']
    prob = np.exp(data['logprob'])

    adv = normalize_adv(
        adv, 
        data, 
        normalize_adv=config.normalize_adv, 
        zero_center=config.zero_center, 
        process_adv=config.process_adv
    )

    target_prob_prime = prob + adv / prob
    tr_prob_prime = prob + config.step_size * adv / prob

    add_probs_to_data(data, target_prob_prime, tr_prob_prime)

def compute_indices(
    idxes, 
    mb_idx, 
    mb_size, 
    n_mbs
):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % n_mbs
    curr_idxes = idxes[start: end]

    return mb_idx, curr_idxes


class AdvantageCalculator:
    def __init__(self, config):
        if config.get('normalize_value', False):
            self.value_rms = RunningMeanStd([0, 1])
        else:
            self.value_rms = None

    def compute_adv(
        self, 
        data, 
        config, 
        last_value=None, 
        value=None, 
        next_value=None,
        reset=None, 
    ):
        if next_value is None and last_value is not None:
            last_value = np.expand_dims(last_value, 0)
            value = np.concatenate([data['value'], last_value], axis=0)
            value = self.denormalize_value(value)
            next_value = value[1:]
            value = value[:-1]
        elif next_value is not None:
            value = data['value']
            value = self.denormalize_value(value)
            next_value = self.denormalize_value(next_value)
        else:
            value = self.denormalize_value(value)
            next_value = value[1:]
            value = value[:-1]

        if config.adv_type == 'gae':
            adv, traj_ret = \
                compute_gae(
                    reward=data['reward'], 
                    discount=data['discount'],
                    value=value,
                    gamma=config.gamma,
                    gae_discount=config.gae_discount,
                    next_value=next_value,
                    reset=reset,
                    norm_adv=False, 
                    mask=data.get('life_mask'),
                    epsilon=config.get('epsilon', 1e-8),
                )
            data['advantage'] = normalize_adv(
                adv, 
                data, 
                normalize_adv=config.normalize_adv, 
                zero_center=config.zero_center_adv, 
                process_adv=config.process_adv,
                adv_clip=config.adv_clip, 
            )
            data['traj_ret'] = self.normalize_value(traj_ret)
            self.update_value_rms(traj_ret)
        elif config.adv_type == 'vtrace':
            pass
        else:
            raise NotImplementedError

    """ Value RMS Operations """    
    def update_value_rms(self, ret, mask=None):
        if self.value_rms is not None:
            self.value_rms.update(ret, mask=mask)

    def normalize_value(self, value, mask=None):
        if self.value_rms is not None:
            value = self.value_rms.normalize(
                value, zero_center=False, mask=mask)
        return value

    def denormalize_value(self, value, mask=None):
        if self.value_rms is not None:
            value = self.value_rms.denormalize(
                value, zero_center=False, mask=mask)
        return value


class TargetPolicyCalculator:
    def compute_target_policy(self, data, config):
        compute_target_probs = {
            'pr': compute_pr_clipped_target, 
            'exp': compute_exp_clipped_target, 
            'adv': compute_adv_clipped_target, 
            'lin': compute_linear_clipped_target, 
            'mix': compute_mixture_target, 
            'step': compute_step_target
        }[config.target_pi.target_type]

        compute_target_probs(data, config.target_pi)

        if 'pi' in data:
            compute_target_pi(data, config.target_pi)


class SamplingKeysExtractor:
    def extract_sampling_keys(self, env_stats: AttrDict, model: Model):
        self.actor_state_keys = tuple([f'actor_{k}' for k in model.actor_state_keys])
        self.value_state_keys = tuple([f'value_{k}' for k in model.value_state_keys])
        self.actor_state_type = model.actor_state_type
        self.value_state_type = model.value_state_type
        self.sample_keys, self.sample_size = self._get_sample_keys_size()
        if env_stats.use_action_mask:
            self.sample_keys.append('action_mask')
        elif 'action_mask' in self.sample_keys:
            self.sample_keys.remove('action_mask')
        if env_stats.use_life_mask:
            self.sample_keys.append('life_mask')
        elif 'life_mask' in self.sample_keys:
            self.sample_keys.remove('life_mask')

    def _get_sample_keys_size(self):
        actor_state_keys = ['actor_h', 'actor_c']
        value_state_keys = ['value_h', 'value_c']
        if self.config.actor_rnn_type or self.config.value_rnn_type: 
            sample_keys = self.config.sample_keys
            sample_size = self.config.sample_size
            if not self.config.actor_rnn_type:
                sample_keys = self._remote_state_keys(
                    sample_keys, 
                    actor_state_keys, 
                )
            if not self.config.value_rnn_type:
                sample_keys = self._remote_state_keys(
                    sample_keys, 
                    value_state_keys, 
                )
        else:
            sample_keys = self._remote_state_keys(
                self.config.sample_keys, 
                actor_state_keys + value_state_keys, 
            )
            if 'mask' in sample_keys:
                sample_keys.remove('mask')
            sample_size = None

        return sample_keys, sample_size

    def _remote_state_keys(self, sample_keys, state_keys):
        for k in state_keys:
            if k in sample_keys:
                sample_keys.remove(k)

        return sample_keys


class Sampler:
    def get_sample(
        self, 
        memory, 
        idxes, 
        sample_keys, 
    ):
        if self.actor_state_type is None and self.value_state_type is None:
            sample = {k: memory[k][idxes] for k in sample_keys}
        else:
            sample = {}
            actor_state = []
            value_state = []
            for k in sample_keys:
                if k in self.actor_state_keys:
                    v = memory[k][idxes, 0]
                    actor_state.append(v.reshape(-1, v.shape[-1]))
                elif k in self.value_state_keys:
                    v = memory[k][idxes, 0]
                    value_state.append(v.reshape(-1, v.shape[-1]))
                else:
                    sample[k] = memory[k][idxes]
            if actor_state:
                sample['actor_state'] = self.actor_state_type(*actor_state)
            if value_state:
                sample['value_state'] = self.value_state_type(*value_state)

        return sample


class LocalBufferBase(
    SamplingKeysExtractor, 
    AdvantageCalculator, 
    TargetPolicyCalculator, 
    Buffer
):
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

        assert not self.config.get('normalize_value', False), 'Do not support normalizing value in local buffer for now'
        AdvantageCalculator.__init__(self, config)

        self._add_attributes(env_stats, model)

    def _add_attributes(self, env_stats, model):
        self.extract_sampling_keys(env_stats, model)

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
        return self._memlen >= self.maxlen

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

        self.compute_adv(data, self.config, last_value)
        self.compute_target_policy(data, self.config)

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


class TurnBasedLocalBufferBase(
    SamplingKeysExtractor, 
    AdvantageCalculator, 
    TargetPolicyCalculator, 
    Buffer
):
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

        assert not self.config.get('normalize_value', False), 'Do not support normalizing value in local buffer for now'
        AdvantageCalculator.__init__(self, config)

        self._add_attributes(env_stats, model)

    def _add_attributes(self, env_stats, model):
        self.extract_sampling_keys(env_stats, model)

        self.maxlen = self.config.n_envs * self.config.n_steps
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
        return self._memlen >= self.maxlen

    def reset(self):
        self._memory = []
        self._memlen = 0
        self._buffers = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self._buff_lens = collections.defaultdict(int)

    def add(self, data: dict, reset):
        eids = data.pop('eid')
        uids = data.pop('uid')
        for i, (eid, uid) in enumerate(zip(eids, uids)):
            if reset[i]:
                assert self._buff_lens[(eid, uid)] == 0, (eid, uid, self._buff_lens[(eid, uid)])
                assert len(self._buffers[(eid, uid)]) == 0, (eid, uid, len(self._buffers[(eid, uid)])) 
            for k, v in data.items():
                self._buffers[(eid, uid)][k].append(v[i])

    def add_reward(self, eids, uids, reward, discount):
        assert len(eids) == len(uids), (eids, uids)
        for i, (eid, uid) in enumerate(zip(eids, uids)):
            if np.any(discount[i] == 0):
                np.testing.assert_equal(discount[i], 0)
                continue
            if not self._buffers[(eid, uid)]:
                continue
            assert 'obs' in self._buffers[(eid, uid)], self._buffers[(eid, uid)]
            self._buffers[(eid, uid)]['reward'].append([reward[i][uid]])
            self._buff_lens[(eid, uid)] += 1

    def finish_episode(self, eids, uids, reward):
        for i, eid in enumerate(eids):
            for uid in uids:
                if self._buffers[(eid, uid)]:
                    self._buffers[(eid, uid)]['reward'].append([reward[i][uid]])
                    self._buff_lens[(eid, uid)] += 1
                    self.merge_episode(eid, uid)
                else:
                    self._reset_buffer(eid, uid)

    def merge_episode(self, eid, uid):
        episode = {k: np.stack(v) for k, v in self._buffers[(eid, uid)].items()}
        episode['discount'] = np.ones_like(episode['reward'], dtype=np.float32)
        episode['discount'][-1] = 0
        epslen = self._buff_lens[(eid, uid)]
        for k, v in episode.items():
            assert v.shape[0] == epslen, (k, v.shape, epslen)

        self.compute_adv(episode, self.config, 
            last_value=np.array([0], np.float32))
        self.compute_target_policy(episode, self.config)

        self._memory.append(episode)
        self._memlen += epslen
        self._reset_buffer(eid, uid)
        return episode

    def retrieve_all_data(self):
        data = batch_dicts(self._memory, np.concatenate)
        for k, v in data.items():
            assert v.shape[0] == self._memlen, (v.shape, self._memlen)
            data[k] = v[-self.maxlen:]
            if self.sample_size is not None:
                data[k] = np.reshape(v, (
                    self.n_envs * self.n_samples, self.sample_size, *v.shape[2:]))
        self.reset()
        return self.runner_id, data, self.maxlen

    def _reset_buffer(self, eid, uid):
        self._buffers[(eid, uid)] = collections.defaultdict(list)
        self._buff_lens[(eid, uid)] = 0


class PPOBufferBase(
    Sampler, 
    SamplingKeysExtractor, 
    AdvantageCalculator, 
    TargetPolicyCalculator, 
    Buffer
):
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

        AdvantageCalculator.__init__(self, config)

        self._add_attributes(env_stats, model)

    def _add_attributes(self, env_stats, model):
        self.use_dataset = self.config.get('use_dataset', False)
        do_logging(f'Is dataset used for data pipeline: {self.use_dataset}', logger=logger)

        self.extract_sampling_keys(env_stats, model)

        self.n_runners = self.config.n_runners
        self.n_envs = self.n_runners * self.config.n_envs
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

    def finish(
        self, 
        *, 
        last_value=None, 
        value=None, 
        next_value=None, 
        reset=None
    ):
        assert self._current_size == self.n_steps, self._current_size
        if value is not None:
            self._memory['value'] = value
        self._memory = {k: np.stack(v) for k, v in self._memory.items()}

        self.compute_adv(
            self._memory, 
            self.config, 
            last_value=last_value, 
            value=value, 
            next_value=next_value, 
            reset=reset, 
        )
        self.compute_target_policy(self._memory, self.config)

        for k, v in self._memory.items():
            assert v.shape[:2] == (self.config.n_steps, self.n_envs), (k, v.shape, (self.config.n_steps, self.config.n_envs))
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
                    [np.concatenate(b[k]) for b in self._buffers if b]
                )[-self.batch_size:]
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
            obs = self._memory['global_state'][curr_idxes]
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

        sample = self.get_sample(
            self._memory, 
            self._curr_idxes, 
            sample_keys,
        )
        sample = self._process_sample(sample)

        return sample

    def _process_sample(self, sample):
        # if 'advantage' in sample and self.config.norm_adv == 'minibatch':
        #     sample['advantage'] = standardize(
        #         sample['advantage'], 
        #         mask=sample.get('life_mask'), 
        #         epsilon=self.config.get('epsilon', 1e-5)
        #     )
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


class LocalBuffer(LocalBufferBase):
    pass


class TurnBasedLocalBuffer(TurnBasedLocalBufferBase):
    pass


class PPOBuffer(PPOBufferBase):
    pass


def create_buffer(config, model, env_stats, **kwargs):
    config = dict2AttrDict(config)
    env_stats = dict2AttrDict(env_stats)
    BufferCls = {
        'ppo': PPOBuffer, 
        'local': LocalBuffer, 
        'tblocal': TurnBasedLocalBuffer
    }[config.type]
    return BufferCls(
        config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )
