import os
import cloudpickle
import numpy as np
import tensorflow as tf

from utility.utils import RunningMeanStd


class RND:
    def __init__(self, model, gamma_int, rms_path):
        self.predictor = model['predictor']
        self.target = model['target']
        self._gamma_int = gamma_int
        axis = (0, 1)
        self._obs_rms = RunningMeanStd(axis=axis, epsilon=1e-4, clip=5)
        self._returns_int = 0
        self._int_return_rms = RunningMeanStd(axis=axis, epsilon=1e-4)
        self._rms_path = rms_path
        self._rms_restored = False

    def compute_int_reward(self, next_obs):
        """ next_obs is expected to be normalized """
        assert len(next_obs.shape) == 5, next_obs.shape
        assert next_obs.dtype == np.float32, next_obs.dtype
        assert next_obs.shape[-1] == 1, next_obs.shape
        reward_int = self._intrinsic_reward(next_obs).numpy()
        returns_int = np.array([self._compute_intrinsic_return(r) for r in reward_int.T])
        self._update_int_return_rms(returns_int)
        reward_int = self._normalize_int_reward(reward_int)
        return reward_int

    @tf.function
    def _intrinsic_reward(self, next_obs):
        target_feat = self.target(next_obs)
        pred_feat = self.predictor(next_obs)
        int_reward = tf.reduce_mean(tf.square(target_feat - pred_feat), axis=-1)
        return int_reward

    def _compute_intrinsic_return(self, reward):
        assert reward.ndim == 1, reward.shape
        # we intend to do the discoutning backwards in time as the future is yet to ocme
        self._returns_int = reward + self._gamma_int * self._returns_int
        return self._returns_int

    def update_obs_rms(self, obs):
        if obs.dtype == np.uint8 and obs.shape[-1] > 1:
            # for stacked frames, we only use
            # the most recent one for rms update
            obs = obs[..., -1:]
        assert len(obs.shape) == 5, obs.shape
        assert obs.dtype == np.uint8, obs.dtype
        self._obs_rms.update(obs)

    def _update_int_return_rms(self, int_return):
        assert len(int_return.shape) == 2
        self._int_return_rms.update(int_return)

    def normalize_obs(self, obs):
        assert len(obs.shape) == 5, obs.shape
        obs_norm = self._obs_rms.normalize(obs[..., -1:])
        assert not np.any(np.isnan(obs_norm))
        return obs_norm

    def _normalize_int_reward(self, reward):
        return self._int_return_rms.normalize(reward, subtract_mean=False)

    def restore(self):
        import os
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._int_return_rms = \
                    cloudpickle.load(f)
            print('RMSs are restored')
            self._rms_restored = True

    def save(self):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump(
                (self._obs_rms, self._int_return_rms), f)

    def rms_restored(self):
        return self._rms_restored
    
    def get_running_stats(self):
        return self._obs_rms.get_stats(), self._int_return_rms.get_stats()
