from abc import ABC
import os
import cloudpickle
import logging

from core.log import *
from core.checkpoint import *


logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """ Restore & save """
    def restore(self):
        """ Restore the latest parameter recorded by ckpt_manager

        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
            ckpt: An instance of tf.train.Checkpoint
            ckpt_path: The directory in which to write checkpoints
            name: optional name for print
        """
        restore(self._ckpt_manager, self._ckpt, self._ckpt_path, self._model_name)
        self.env_step = self._env_step.numpy()
        self.train_step = self._train_step.numpy()

    def save(self, print_terminal_info=False):
        """ Save Model
        
        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
        """
        self._env_step.assign(self.env_step)
        self._train_step.assign(self.train_step)
        save(self._ckpt_manager, print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        save_config(self._root_dir, self._model_name, config)

    def log(self, step, prefix=None, print_terminal_info=True):
        log(self._logger, self._writer, self._model_name, prefix=prefix, 
            step=step, print_terminal_info=print_terminal_info)

    def log_stats(self, stats, print_terminal_info=True):
        log_stats(self._logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
            
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def store(self, **kwargs):
        store(self._logger, **kwargs)

    def get_raw_item(self, key):
        return get_raw_item(self._logger, key)

    def get_item(self, key, mean=True, std=False, min=False, max=False):
        return get_item(self._logger, key, mean=mean, std=std, min=min, max=max)

    def get_raw_stats(self):
        return get_raw_stats(self._logger)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        return get_stats(self._logger, mean=mean, std=std, min=min, max=max)

    def print_construction_complete(self):
        pwc(f'{self.name.upper()} is constructed...', color='cyan')


class RMSBaseAgent(BaseAgent):
    def __init__(self):
        from utility.utils import RunningMeanStd
        self._normalized_axis = getattr(self, '_normalized_axis', (0, 1))
        self._normalize_obs = getattr(self, '_normalize_obs', False)
        self._normalize_reward = getattr(self, '_normalize_reward', True)
        self._normalize_reward_with_reversed_return = \
            getattr(self, '_normalize_reward_with_reversed_return', True)
        
        axis = tuple(self._normalized_axis)
        self._obs_rms = self._normalize_obs \
            and RunningMeanStd(axis)
        self._reward_rms = self._normalize_reward \
            and RunningMeanStd(axis)
        if self._normalize_reward_with_reversed_return:
            self._reverse_return = 0
        else:
            self._reverse_return = -np.inf
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'
        logger.info(f'Reward normalization: {self._normalize_reward}')
        logger.info(f'Reward normalization with reversed return: {self._normalize_reward_with_reversed_return}')

    """ Functions for running mean and std """
    def get_running_stats(self):
        stats = ()
        if self._normalize_obs:
            stats += self._obs_rms.get_stats()
        if self._normalize_reward:
            stats += self._reward_rms.get_stats()
        return stats

    @property
    def is_obs_or_reward_normalized(self):
        return self._obs_rms or self._reward_rms

    def update_obs_rms(self, obs):
        if self._normalize_obs:
            # if obs.dtype == np.uint8 and obs.shape[-1] > 1:
            #     # for stacked frames, we only use
            #     # the most recent one for rms update
            #     obs = obs[..., -1:]
            self._obs_rms.update(obs)

    def update_reward_rms(self, reward, discount=None):
        if self._normalize_reward:
            assert len(reward.shape) == len(self._normalized_axis), (reward.shape, self._normalized_axis)
            if self._normalize_reward_with_reversed_return:
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
                    f"Normalizing rewards with reversed return requires environment's discount(i.e., 1-done) signals"
                ret = backward_discounted_sum(self._reverse_return, reward, discount, self._gamma)
                self._reverse_return = ret[:, -1]
                self._reward_rms.update(ret)
            else:
                self._reward_rms.update(reward)

    def normalize_obs(self, obs):
        return self._obs_rms.normalize(obs) \
            if self._normalize_obs else obs

    def normalize_reward(self, reward):
        if self._normalize_reward:
            return self._reward_rms.normalize(reward, subtract_mean=False)
        else:
            return reward

    def restore(self):
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._reward_rms = cloudpickle.load(f)
        super().restore()

    def save(self, print_terminal_info=False):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump((self._obs_rms, self._reward_rms), f)
        super().save(print_terminal_info=print_terminal_info)


def backward_discounted_sum(prev_ret, reward, discount, gamma,):
    _nenv, nstep = reward.shape
    ret = np.zeros_like(reward)
    for t in range(nstep):
        prev_ret = ret[:, t] = reward[:, t] + gamma * prev_ret
        prev_ret *= discount[:, t]
    return ret
