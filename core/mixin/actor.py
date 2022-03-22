import os
import cloudpickle
import collections
import logging
import numpy as np
from typing import Dict, List, Tuple, Union

from core.log import do_logging
from core.typing import ModelPath
from utility.rms import RunningMeanStd, combine_rms
from utility.utils import dict2AttrDict

logger = logging.getLogger(__name__)


RMSStats = collections.namedtuple('RMSStats', 'obs reward')


def combine_rms_stats(rms_stats1: RMSStats, rms_stats2: RMSStats):
    obs_rms = {}
    for k in rms_stats1.obs.keys():
        obs_rms[k] = combine_rms(rms_stats1.obs[k], rms_stats2.obs[k])
        # for n, rms in zip([f'{k}_before', f'{k}_new', f'{k}_after'], [rms_stats1.obs[k], rms_stats2.obs[k], obs_rms[k]]):
        #     for i, (m, v) in enumerate(zip(*rms[:2])):
        #         do_logging('combine rms stats: {n}, {i}, mean, {m}, var, {v}', logger=logger)
    if rms_stats1.reward:
        reward_rms = combine_rms(rms_stats1.reward, rms_stats2.reward)
    else:
        reward_rms = None
    return RMSStats(obs_rms, reward_rms)


class RMS:
    def __init__(self, config: dict, name='rms'):
        # by default, we update reward stats once every N steps 
        # so we normalize along the first two axis
        config = dict2AttrDict(config)
        self.name = name
        self._gamma = config['gamma']
        self._reward_normalized_axis = tuple(
            config.get('reward_normalized_axis', (0, 1)))
        # by default, we update obs stats every step so we normalize along the first axis
        self._obs_normalized_axis = tuple(
            config.get('obs_normalized_axis', (0,)))
        self._normalize_obs = config.get('normalize_obs', False)
        self._normalize_reward = config.get('normalize_reward', False)
        self._normalize_reward_with_return = \
            config.get('normalize_reward_with_return', 'backward')
        self._update_reward_rms_in_time = config.get('update_reward_rms_in_time', True)
        assert self._normalize_reward_with_return in ('backward', 'forward', None)
        if self._update_reward_rms_in_time:
            assert self._normalize_reward_with_return == 'backward', self._normalize_reward
        
        self._obs_names = config.get('obs_names', ['obs'])
        self._obs_rms: Dict[str, RunningMeanStd] = {}
        if self._normalize_obs:
            # we use dict to track a set of observation features
            for k in self._obs_names:
                self._obs_rms[k] = RunningMeanStd(
                    self._obs_normalized_axis, 
                    clip=config.get('obs_clip', 5), 
                    name=f'{k}_rms', ndim=1)
        self._reward_rms = self._normalize_reward \
            and RunningMeanStd(self._reward_normalized_axis, 
                clip=config.get('rew_clip', 10), 
                name='reward_rms', ndim=0)
        if self._normalize_reward_with_return is not None:
            self._return = 0
        else:
            self._return = -np.inf

        if 'model_path' in config:
            self.reset_path(config.model_path)
        else:
            self._rms_path = None

        do_logging(
            f'Observation normalization: {self._normalize_obs}', logger=logger)
        do_logging(
            f'Normalized observation names: {self._obs_names}', logger=logger)
        do_logging(
            f'Reward normalization: {self._normalize_reward}', logger=logger)
        do_logging(
            f'Reward normalization with return: {self._normalize_reward_with_return}', 
            logger=logger)

    """ Attributes """
    @property
    def obs_names(self):
        return self._obs_names

    @property
    def is_obs_or_reward_normalized(self):
        return self._normalize_obs or self._normalize_reward
    
    @property
    def is_obs_normalized(self):
        return self._normalize_obs

    @property
    def is_reward_normalized(self):
        return self._normalize_reward

    """ Processing Data with RMS """
    def process_obs_with_rms(
        self, 
        inp: Union[dict, Tuple[str, np.ndarray]], 
        name: str=None,
        update_rms: bool=False, 
        mask=None
    ):
        """ Do obs normalization if required
        
        Args:
            inp: input to the model, including obs
            mask: life mask, implying if the agent is still alive,
                useful for multi-agent environments, where 
                some agents might be dead before others.
        """
        if name is None:
            for k in self._obs_names:
                if k not in inp:
                    continue
                if update_rms:
                    self.update_obs_rms(inp, k, mask=mask)
                # mask is important here as the value function still matters
                # even after the agent is dead
                inp[k] = self.normalize_obs(inp, k, mask=mask)
        else:
            inp[name] = self.normalize_obs(inp, name, mask)
        return inp

    def process_reward_with_rms(
        self,
        reward: np.ndarray, 
        update_rms: bool=False, 
        discount: np.ndarray=None,
        mask=None
    ):
        if update_rms:
            self.update_reward_rms(reward, discount, mask)
        reward = self.normalize_reward(reward, mask)
        return reward

    def normalize_obs(self, obs, name='obs', mask=None):
        """ Normalize obs using obs RMS """
        return self._obs_rms[name].normalize(obs[name], mask=mask) \
            if self._normalize_obs else obs

    def normalize_reward(self, reward, mask=None):
        """ Normalize obs using reward RMS """
        return self._reward_rms.normalize(reward, zero_center=False, mask=mask) \
            if self._normalize_reward else reward

    """ RMS Access & Override """
    def reset_rms_stats(self):
        for rms in self._obs_rms.values():
            rms.reset_rms_stats()
        if self._reward_rms:
            self._reward_rms.reset_rms_stats()

    def set_rms_stats(self, rms_stats: RMSStats):
        if rms_stats.obs:
            for k, v in rms_stats.obs.items():
                self._obs_rms[k].set_rms_stats(*v)
        if rms_stats.reward:
            self._reward_rms.set_rms_stats(*rms_stats.reward)

    def get_rms_stats(self):
        return RMSStats(self.get_obs_rms_stats(), self.get_rew_rms_stats())

    def get_obs_rms_stats(self):
        obs_rms = {k: v.get_rms_stats() for k, v in self._obs_rms.items()} \
            if self._normalize_obs else {}
        return obs_rms

    def get_rew_rms_stats(self):
        rew_rms = self._reward_rms.get_rms_stats() if self._normalize_reward else None
        return rew_rms

    """ RMS Update """
    def update_all_rms(self, data, obs_mask=None, reward_mask=None, axis=None):
        self.update_obs_rms(data, mask=obs_mask, axis=axis)
        self.update_reward_rms(data['reward'], data['discount'], 
            mask=reward_mask, axis=axis)

    def update_obs_rms(self, obs, name=None, mask=None, axis=None):
        if self._normalize_obs:
            if name is None:
                for k in self._obs_names:
                    assert not obs[k].dtype == np.uint8, f'Unexpected normalization on {name} of type uint8.'
                    self._obs_rms[k].update(obs[k], mask, axis=axis)
            else:
                assert not obs[name].dtype == np.uint8, f'Unexpected normalization on {name} of type uint8.'
                self._obs_rms[name].update(obs[name], mask=mask, axis=axis)

    def update_reward_rms(self, reward, discount=None, mask=None, axis=None):
        def forward_discounted_sum(next_ret, reward, discount, gamma):
            assert reward.shape == discount.shape, (reward.shape, discount.shape)
            # we assume the sequential dimension is at the first axis
            nstep = reward.shape[0]
            ret = np.zeros_like(reward)
            for t in reversed(range(nstep)):
                ret[t] = next_ret = reward[t] + gamma * discount[t] * next_ret
            return next_ret, ret

        def backward_discounted_sum(prev_ret, reward, discount, gamma):
            """ Compute the discounted sum of rewards in the reverse order """
            assert reward.shape == discount.shape, (reward.shape, discount.shape)
            if self._update_reward_rms_in_time:
                prev_ret = reward + gamma * prev_ret
                ret = prev_ret.copy()
                prev_ret *= discount
                return prev_ret, ret
            else:
                # we assume the sequential dimension is at the first axis
                nstep = reward.shape[0]
                ret = np.zeros_like(reward)
                for t in range(nstep):
                    ret[t] = prev_ret = reward[t] + gamma * prev_ret
                    prev_ret *= discount[t]
                return prev_ret, ret

        if self._normalize_reward:
            assert len(reward.shape) == len(self._reward_normalized_axis), \
                (reward.shape, self._reward_normalized_axis)
            if self._normalize_reward_with_return == 'backward':
                """
                Pseudocode can be found in https://arxiv.org/pdf/1811.02553.pdf
                section 9.3 (which is based on our Baselines code, haha)
                Motivation is that we'd rather normalize the returns = sum of future rewards,
                but we haven't seen the future yet. So we assume that the time-reversed rewards
                have similar statistics to the rewards, and normalize the time-reversed rewards.

                Quoted from
                https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/reward_normalizer.py
                Yeah, you may not find the pseudocode. That's why I quote:-)
                """
                assert discount is not None, \
                    f"Normalizing rewards with backward return requires environment's reset signals"
                assert reward.ndim == discount.ndim == len(self._reward_rms.axis), \
                    (reward.shape, discount.shape, self._reward_rms.axis)
                self._return, ret = backward_discounted_sum(
                    self._return, reward, discount, self._gamma)
                self._reward_rms.update(ret, mask=mask, axis=axis)
            elif self._normalize_reward_with_return == 'forward':
                self._return, ret = forward_discounted_sum(
                    self._return, reward, discount, self._gamma)
                self._reward_rms.update(ret, mask=mask, axis=axis)
            elif self._normalize_reward_with_return == False:
                self._reward_rms.update(reward, mask=mask, axis=axis)
            else:
                raise ValueError(f"Invalid option: {self._normalize_reward_with_return}")

    def update_rms_from_stats_list(self, rms_stats_list: List[RMSStats]):
        for rms_stats in rms_stats_list:
            self.update_rms_from_stats(rms_stats)

    def update_rms_from_stats(self, rms_stats: RMSStats):
        if rms_stats.obs is not None:
            self.update_obs_rms_from_stats(rms_stats.obs)
        if rms_stats.reward is not None:
            self.update_rew_rms_from_stats(rms_stats.reward)

    def update_obs_rms_from_stats(self, obs_rms):
        for k, v in obs_rms.items():
            self._obs_rms[k].update_from_moments(
                batch_mean=v.mean,
                batch_var=v.var,
                batch_count=v.count)

    def update_rew_rms_from_stats(self, rew_rms):
        self._reward_rms.update_from_moments(
            batch_mean=rew_rms.mean,
            batch_var=rew_rms.var,
            batch_count=rew_rms.count)

    """ Checkpoint Operations """
    def restore_rms(self):
        if self._rms_path is None:
            raise RuntimeError('rms path is not configured.')
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._reward_rms, self._return = cloudpickle.load(f)
                do_logging(f'rms stats are restored from {self._rms_path}', logger=logger)
            if isinstance(self._reward_rms, RunningMeanStd):
                assert self._reward_rms.axis == self._reward_normalized_axis, \
                    (self._reward_rms.axis, self._reward_normalized_axis)
            if self._obs_rms is None:
                self._obs_rms = {}
            for v in self._obs_rms.values():
                assert v.axis == self._obs_normalized_axis, (v.axis, self._obs_normalized_axis)
    
    def save_rms(self):
        if self._rms_path is None:
            raise RuntimeError('rms path is not configured.')
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump((self._obs_rms, self._reward_rms, self._return), f)

    def reset_path(self, model_path: ModelPath):
        self._rms_path = '/'.join([*model_path, f'{self.name}.pkl'])
