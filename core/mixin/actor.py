import collections
import logging
import numpy as np
from typing import Dict, List, Tuple, Union

from core.ckpt.pickle import save, restore
from core.log import do_logging
from core.typing import ModelPath, AttrDict
from tools.display import print_dict, print_dict_info
from tools.rms import RunningMeanStd, combine_rms, StatsWithVarCount

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


def rms2dict(rms: RMSStats):
    stats = {}
    if rms.obs:
        for k, v in rms.obs.items():
            for kk, vv in v._asdict().items():
                stats[f'aux/{k}/{kk}'] = vv
    if rms.reward:
        for k, v in rms.reward._asdict().items():
            stats[f'aux/reward/{k}'] = v

    return stats


class RewardRunningMeanStd:
    def __init__(self, config, name='reward_rms'):
        self.name = name
        self._filedir = '/'.join(config.model_path)

        self._gamma = config['gamma']
        self._reward_normalized_axis = tuple(
            config.setdefault('reward_normalized_axis', (0, 1)))
        self._normalize_reward = config.setdefault('normalize_reward', False)
        self._normalize_reward_with_return = \
            config.setdefault('normalize_reward_with_return', 'backward')
        self._update_reward_rms_in_time = config.setdefault('update_reward_rms_in_time', False)
        assert self._normalize_reward_with_return in ('backward', 'forward', None)
        if self._update_reward_rms_in_time:
            assert self._normalize_reward_with_return == 'backward', self._normalize_reward
        self.rms = RunningMeanStd(
            self._reward_normalized_axis, 
            clip=config.setdefault('rew_clip', 10), 
            name=name, 
            ndim=config.setdefault("reward_normalized_ndim", 0)
        )
        print_dict(config, prefix=name)

        self.reset_return()

    @property
    def is_normalized(self):
        return self._normalize_reward
    
    def reset_return(self):
        if self._normalize_reward_with_return is not None:
            self._return = 0
        else:
            self._return = -np.inf

    def update(self, reward, discount=None, mask=None, axis=None):
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
                assert discount is not None, f"Normalizing rewards with backward return requires environment's reset signals"
                assert reward.ndim == discount.ndim, (reward.shape, discount.shape)
                self._return, ret = backward_discounted_sum(
                    self._return, reward, discount, self._gamma)
                self.rms.update(ret, mask=mask, axis=axis)
            elif self._normalize_reward_with_return == 'forward':
                self._return, ret = forward_discounted_sum(
                    self._return, reward, discount, self._gamma)
                self.rms.update(ret, mask=mask, axis=axis)
            elif self._normalize_reward_with_return == False:
                self.rms.update(reward, mask=mask, axis=axis)
            else:
                raise ValueError(f"Invalid option: {self._normalize_reward_with_return}")

    def update_from_stats(self, rms: StatsWithVarCount):
        self.rms.update_from_moments(
            batch_mean=rms.mean,
            batch_var=rms.var,
            batch_count=rms.count
        )

    def normalize(self, reward, mask=None):
        """ Normalize obs using reward RMS """
        if self._normalize_reward:
            return self.rms.normalize(reward, zero_center=False, mask=mask)
        else:
            return reward

    def process(
        self,
        reward: np.ndarray, 
        discount: np.ndarray=None,
        update_rms: bool=False, 
        mask=None
    ):
        if update_rms:
            self.update(reward, discount, mask)
        reward = self.normalize(reward, mask)
        return reward

    """ RMS Access & Override """
    def reset_rms_stats(self):
        self.rms.reset_rms_stats()

    def get_rms_stats(self):
        return self.rms.get_rms_stats()

    def set_rms_stats(self, rms_stats: StatsWithVarCount):
        self.rms.set_rms_stats(*rms_stats)

    def print_rms(self):
        print_dict_info(self.rms.get_rms_stats(), prefix=self.name)

    """ Checkpoint Operations """
    def restore_rms(self):
        if self._filedir is None:
            raise RuntimeError('rms path is not configured.')
        self.rms = restore(
            filedir=self._filedir, 
            filename=self.name, 
            default=self.rms
        )
    
    def save_rms(self):
        if self._filedir is None:
            raise RuntimeError('rms path is not configured.')
        save(self.rms, filedir=self._filedir, filename=self.name)

    def reset_path(self, model_path: ModelPath):
        self._filedir = '/'.join([*model_path, f'{self.name}.pkl'])


class ObsRunningMeanStd:
    def __init__(self, config: dict, name='obs_rms'):
        self.name = name
        self._filedir = '/'.join(config.model_path)

        self._obs_normalized_axis = tuple(
            config.get('obs_normalized_axis', (0,)))
        self._normalize_obs = config.get('normalize_obs', False)

        self._obs_names = config.get('obs_names', ['obs'])
        self._masked_names = config.get('masked_names', ['obs'])
        self.rms: Dict[str, RunningMeanStd] = AttrDict()
        if self._normalize_obs:
            # we use dict to track a set of observation features
            for k in self._obs_names:
                self.rms[k] = RunningMeanStd(
                    self._obs_normalized_axis, 
                    clip=config.setdefault('obs_clip', 5), 
                    name=f'{k}_rms', 
                    ndim=config.setdefault("obs_normalized_ndim", 1)
                )
        print_dict(config, prefix=name)

    @property
    def obs_names(self):
        return self._obs_names

    @property
    def is_normalized(self):
        return self._normalize_obs

    """ Processing Data with RMS """
    def update(self, obs, name=None, mask=None, axis=None):
        if self._normalize_obs:
            if not isinstance(obs, dict):
                if name is None:
                    name = 'obs'
                if name not in self._masked_names:
                    mask = None
                self.rms[name].update(obs, mask=mask, axis=axis)
            else:
                if name is None:
                    for k in self._obs_names:
                        assert not obs[k].dtype == np.uint8, f'Unexpected normalization on {name} of type uint8.'
                        rms_mask = mask if name in self._masked_names else None
                        self.rms[k].update(obs[k], mask=rms_mask, axis=axis)
                else:
                    assert not obs[name].dtype == np.uint8, f'Unexpected normalization on {name} of type uint8.'
                    if name not in self._masked_names:
                        mask = None
                    self.rms[name].update(obs[name], mask=mask, axis=axis)

    def update_from_stats(self, rms: dict):
        for k, v in rms.items():
            self.rms[k].update_from_moments(
                batch_mean=v.mean,
                batch_var=v.var,
                batch_count=v.count
            )

    def normalize(self, obs, name='obs', mask=None):
        """ Normalize obs using obs RMS """
        if isinstance(obs, dict):
            obs = obs[name]
        if self._normalize_obs:
            return self.rms[name].normalize(obs, mask=mask)
        else:
            return obs

    def process(
        self, 
        inp: Union[dict, Tuple[str, np.ndarray]], 
        name: str=None,
        update_rms: bool=False, 
        mask=None
    ):
        if not self._normalize_obs:
            return inp
        if name is None:
            for k in self._obs_names:
                if k not in inp:
                    continue
                if update_rms:
                    self.update(inp, k, mask=mask)
                # mask is important here as the value function still matters
                # even after the agent is dead
                inp[k] = self.normalize(inp, k, mask=mask)
        else:
            inp[name] = self.normalize(inp, name, mask=mask)
        return inp

    """ RMS Access & Override """
    def reset_rms_stats(self):
        for rms in self.rms.values():
            rms.reset_rms_stats()
        if self.rms:
            self.rms.reset_rms_stats()

    def get_rms_stats(self, with_count=True, return_std=False, return_dict=False):
        rms = AttrDict()
        if self._normalize_obs:
            for k, v in self.rms.items():
                rms[k] = v.get_rms_stats(with_count=with_count, return_std=return_std) 
                if return_dict:
                    rms[k] = rms[k]._asdict()

        return rms

    def set_rms_stats(self, rms_stats: StatsWithVarCount):
        if rms_stats:
            for k, v in rms_stats.obs.items():
                self.rms[k].set_rms_stats(*v)

    def print_rms(self):
        for k, v in self.rms.items():
            print_dict_info(v.get_rms_stats(), prefix=f'{self.name}/{k}')

    """ Checkpoint Operations """
    def restore_rms(self):
        if self._filedir is None:
            raise RuntimeError('rms path is not configured.')
        self.rms = restore(
            filedir=self._filedir, 
            filename=self.name, 
            default=self.rms
        )

    def save_rms(self):
        if self._filedir is None:
            raise RuntimeError('rms path is not configured.')
        save(self.rms, filedir=self._filedir, filename=self.name)

    def reset_path(self, model_path: ModelPath):
        self._filedir = '/'.join([*model_path, f'{self.name}.pkl'])


class RMS:
    def __init__(self, config: dict, name='rms'):
        self.obs_rms = ObsRunningMeanStd(config.obs)
        self.reward_rms = RewardRunningMeanStd(config.reward)

    @property
    def is_obs_or_reward_normalized(self):
        return self.obs_rms.is_normalized or self.reward_rms.is_normalized

    """ RMS Access & Override """
    def reset_rms_stats(self):
        self.obs_rms.reset_rms_stats()
        self.reward_rms.reset_rms_stats()

    def get_rms_stats(self):
        return RMSStats(self.obs_rms.get_rms_stats(), self.reward_rms.get_rms_stats())

    def set_rms_stats(self, rms_stats: RMSStats):
        self.obs_rms.set_rms_stats(*rms_stats.obs)
        self.reward_rms.set_rms_stats(*rms_stats.reward)

    """ RMS Update """
    def update_all_rms(self, data, obs_mask=None, reward_mask=None, axis=None):
        self.update_obs_rms(data, mask=obs_mask, axis=axis)
        self.update_reward_rms(data['reward'], data['discount'], 
            mask=reward_mask, axis=axis)

    def update_from_stats_list(self, rms_stats_list: List[RMSStats]):
        for rms_stats in rms_stats_list:
            self.update_from_stats(rms_stats)

    def update_from_stats(self, rms_stats: RMSStats):
        if rms_stats.obs is not None:
            self.obs_rms.update_from_stats(rms_stats.obs)
        if rms_stats.reward is not None:
            self.reward_rms.update_from_stats(rms_stats.reward)

    """ Checkpoint Operations """
    def restore_rms(self):
        self.obs_rms.restore_rms()
        self.reward_rms.restore_rms()

    def save_rms(self):
        self.obs_rms.save_rms()
        self.reward_rms.save_rms()

    def reset_path(self, model_path: ModelPath):
        self.obs_rms.reset_path(model_path)
        self.reward_rms.reset_path(model_path)
