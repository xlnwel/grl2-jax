import logging
import cloudpickle
import numpy as np
import tensorflow as tf

from utility.utils import Every
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.base import BaseAgent
from core.decorator import step_track
from core.optimizer import Optimizer
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


logger = logging.getLogger(__name__)

class PPOBase(BaseAgent):
    def __init__(self, dataset, env):
        from env.wrappers import get_wrapper_by_name
        from utility.utils import RunningMeanStd
        self.dataset = dataset
        axis = None if get_wrapper_by_name(env, 'Env') else 0
        self._obs_rms = RunningMeanStd(axis=axis)
        self._reward_rms = RunningMeanStd(axis=axis)
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

        # optimizer
        if getattr(self, 'schedule_lr', False):
            self._lr = TFPiecewiseSchedule(
                [(300, self._lr), (1000, 5e-5)])
        self._optimizer = Optimizer(
            self._optimizer, self.ac, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)

    """ Functions for running mean and std """
    @property
    def is_obs_or_reward_normalized(self):
        return self._normalize_obs and self._normalize_reward

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

    """ Standard PPO functions """
    def reset_states(self, state=None):
        raise NotImplementedError

    def get_states(self):
        raise NotImplementedError

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            for j in range(1, self.N_MBS+1):
                data = self.dataset.sample()
                value = data['value']
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.learn(**data)

                terms = {k: v.numpy() for k, v in terms.items()}

                terms['value'] = np.mean(value)
                kl, p_clip_frac, v_clip_frac = \
                    terms['kl'], terms['p_clip_frac'], terms['v_clip_frac']
                for k in ['kl', 'p_clip_frac', 'v_clip_frac']:
                    del terms[k]

                self.store(**terms)
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                        break
            if getattr(self, '_max_kl', None) and kl > self._max_kl:
                logger.info(f'{self._model_name}: Eearly stopping after '
                    f'{i*self.N_MBS+j} update(s) due to reaching max kl.',
                    f'Current kl={kl:.3g}')
                break
        self.store(kl=kl, p_clip_frac=p_clip_frac, v_clip_frac=v_clip_frac)
        if not isinstance(self._lr, float):
            step = tf.cast(self._env_step, tf.float32)
            self.store(lr=self._lr(step))
        
        if self._to_summary(step):
            self.summary(data, terms)

        return i * self.N_MBS + j

    def summary(self, data, terms):
        tf.summary.histogram('traj_ret', data['traj_ret'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, traj_ret, value, advantage, logpi, mask=None, state=None):
        old_value = value
        with tf.GradientTape() as tape:
            if mask is None:
                act_dist, value = self.ac(obs, return_value=True)
            else:
                act_dist, value, state = self.ac(obs, state, mask=mask, return_value=True)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            ppo_loss, entropy, kl, p_clip_frac = compute_ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self._clip_range)

            policy_loss = (ppo_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss
            ac_loss = policy_loss + value_loss

        terms = dict(
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            v_clip_frac=v_clip_frac,
            ppo_loss=ppo_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value)
        )
        terms['ac_norm'] = self._optimizer(tape, ac_loss)

        return terms
