import os
import cloudpickle
import collections
import logging
import numpy as np
from typing import Tuple, Union

from core.log import do_logging
from utility.rms import RunningMeanStd

logger = logging.getLogger(__name__)


RMSStats = collections.namedtuple('RMSStats', 'obs reward')


class RMS:
    def __init__(self, config: dict, name='rms'):
        # by default, we update reward stats once every N steps 
        # so we normalize along the first two axis
        self._gamma = config['gamma']
        self._reward_normalized_axis = tuple(
            config.get('reward_normalized_axis', (0, 1)))
        # by default, we update obs stats every step so we normalize along the first axis
        self._obs_normalized_axis = tuple(
            config.get('obs_normalized_axis', (0,)))
        self._normalize_obs = config.get('normalize_obs', False)
        self._normalize_reward = config.get('normalize_reward', False)
        self._normalize_reward_with_return = \
            config.get('normalize_reward_with_return', 'reversed')
        assert self._normalize_reward_with_return in ('reversed', 'forward', None)
        
        self._obs_names = config.get('obs_names', ['obs'])
        self._obs_rms = {}
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
        if 'root_dir' in config:
            self._rms_path = f'{config["root_dir"]}/{config["model_name"]}/{name}.pkl'
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
        if self._normalize_reward_with_return:
            do_logging(
                f"Reward normalization axis: {'1st' if self._normalize_reward_with_return == 'forward' else '2nd'}", 
                logger=logger)

    def process_obs_with_rms(self, 
                             inp: Union[dict, Tuple[str, np.ndarray]], 
                             update_rms: bool=True, 
                             mask=None):
        """ Do obs normalization if required
        
        Args:
            inp: input to the model, including obs
            mask: life mask, implying if the agent is still alive,
                useful for multi-agent environments, where 
                some agents might be dead before others.
        """
        if isinstance(inp, dict):
            for k in self._obs_names:
                if k not in inp:
                    continue
                v = inp[k]
                if update_rms:
                    self.update_obs_rms(v, k, mask=mask)
                # mask is important here as the value function still matters
                # even after the agent is dead
                inp[k] = self.normalize_obs(v, k, mask=mask)
        else:
            k, inp = inp
            inp = self.normalize_obs(inp, k, mask)
        return inp

    def set_rms_stats(self, obs_rms={}, rew_rms=None):
        if obs_rms:
            for k, v in obs_rms.items():
                self._obs_rms[k].set_rms_stats(*v)
        if rew_rms:
            self._reward_rms.set_rms_stats(*rew_rms)

    def get_rms_stats(self):
        return RMSStats(self.get_obs_rms_stats(), self.get_rew_rms_stats())

    def get_obs_rms_stats(self):
        obs_rms = {k: v.get_rms_stats() for k, v in self._obs_rms.items()} \
            if self._normalize_obs else {}
        return obs_rms

    def get_rew_rms_stats(self):
        rew_rms = self._reward_rms.get_rms_stats() if self._normalize_reward else ()
        return rew_rms

    @property
    def is_obs_or_reward_normalized(self):
        return self._normalize_obs or self._normalize_reward
    
    @property
    def is_obs_normalized(self):
        return self._normalize_obs

    @property
    def is_reward_normalized(self):
        return self._normalize_reward

    def update_obs_rms(self, obs, name='obs', mask=None):
        if self._normalize_obs:
            if obs.dtype == np.uint8 and \
                    getattr(self, '_image_normalization_warned', False):
                logger.warning('Image observations are normalized. Make sure you intentionally do it.')
                self._image_normalization_warned = True
            self._obs_rms[name].update(obs, mask=mask)

    def update_reward_rms(self, reward, discount=None, mask=None):
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
            if reward.ndim == 1:
                prev_ret = reward + gamma * prev_ret
                ret = prev_ret.copy()
                prev_ret *= discount
                return prev_ret, ret
            else:
                # we assume the sequential dimension is at the second axis
                nstep = reward.shape[1]
                ret = np.zeros_like(reward)
                for t in range(nstep):
                    ret[:, t] = prev_ret = reward[:, t] + gamma * prev_ret
                    prev_ret *= discount[:, t]
                return prev_ret, ret

        if self._normalize_reward:
            assert len(reward.shape) == len(self._reward_normalized_axis), \
                (reward.shape, self._reward_normalized_axis)
            if self._normalize_reward_with_return == 'reversed':
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
                    f"Normalizing rewards with reversed return requires environment's reset signals"
                assert reward.ndim == discount.ndim == len(self._reward_rms.axis), \
                    (reward.shape, discount.shape, self._reward_rms.axis)
                self._return, ret = backward_discounted_sum(
                    self._return, reward, discount, self._gamma)
                self._reward_rms.update(ret, mask=mask)
            elif self._normalize_reward_with_return == 'forward':
                self._return, ret = forward_discounted_sum(
                    self._return, reward, discount, self._gamma)
                self._reward_rms.update(ret, mask=mask)
            elif self._normalize_reward_with_return == False:
                self._reward_rms.update(reward, mask=mask)
            else:
                raise ValueError(f"Invalid option: {self._normalize_reward_with_return}")

    def normalize_obs(self, obs, name='obs', mask=None):
        """ Normalize obs using obs RMS """
        return self._obs_rms[name].normalize(obs, mask=mask) \
            if self._normalize_obs else obs

    def normalize_reward(self, reward, mask=None):
        """ Normalize obs using reward RMS """
        return self._reward_rms.normalize(reward, zero_center=False, mask=mask) \
            if self._normalize_reward else reward

    def update_from_rms_stats(self, obs_rms, rew_rms):
        for k, v in obs_rms.items():
            if v:
                self._obs_rms[k].update_from_moments(
                    batch_mean=v.mean,
                    batch_var=v.var,
                    batch_count=v.count)
        if rew_rms:
            self._reward_rms.update_from_moments(
                batch_mean=rew_rms.mean,
                batch_var=rew_rms.var,
                batch_count=rew_rms.count)

    def restore_rms(self):
        if self._rms_path is None:
            raise RuntimeError('rms path is not configured.')
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._reward_rms, self._return = cloudpickle.load(f)
                do_logging(f'rms stats are restored from {self._rms_path}', logger=logger)
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
