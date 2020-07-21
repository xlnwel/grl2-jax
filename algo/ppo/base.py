import cloudpickle
import numpy as np
import tensorflow as tf

from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer


class PPOBase(BaseAgent):
    def __init__(self, dataset, env):
        from env.wrappers import get_wrapper_by_name
        from utility.utils import RunningMeanStd
        self.dataset = dataset
        axis = None if get_wrapper_by_name(env, 'Env') else 0
        self._obs_rms = RunningMeanStd(axis=axis)
        self._reward_rms = RunningMeanStd(axis=axis)
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

    def update_obs_rms(self, obs):
        assert len(obs.shape) in (2, 4)
        if self._normalize_obs:
            if obs.dtype == np.uint8 and obs.shape[-1] > 1:
                # for stacked frames, we only use
                # the most recent one for rms update
                obs = obs[..., -1:]
            self._obs_rms.update(obs)

    def update_reward_rms(self, reward):
        assert len(reward.shape) == 1
        if self._normalize_reward:
            self._reward_rms.update(reward)

    def normalize_obs(self, obs, update_rms=False):
        if self._normalize_obs:
            return self._obs_rms.normalize(obs)
        else:
            return np.array(obs, copy=False)

    def normalize_reward(self, reward):
        if self._normalize_reward:
            return self._reward_rms.normalize(reward, subtract_mean=False)
        else:
            return reward

    def restore(self):
        import os
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._reward_rms = cloudpickle.load(f)
        super().restore()

    def save(self, print_terminal_info=False):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump((self._obs_rms, self._reward_rms), f)
        super().save(print_terminal_info=print_terminal_info)