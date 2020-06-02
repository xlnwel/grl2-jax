import cloudpickle
import numpy as np
import tensorflow as tf

from env.wrappers import get_wrapper_by_name
from utility.utils import RunningMeanStd
from utility.display import pwc
from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class RNDBase(BaseAgent):
    def __init__(self, buffer, env):
        from env.wrappers import get_wrapper_by_name
        from utility.utils import RunningMeanStd
        self.buffer = buffer
        axis = 0 if get_wrapper_by_name(env, 'Env') else (0, 1)
        self._obs_rms = RunningMeanStd(axis=axis, clip=5)
        self._int_reward_rms = RunningMeanStd(axis=axis)
        self._ext_reward_rms = RunningMeanStd(axis=axis)
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

    def compute_int_reward(self, next_obs):
        """ next_obs is expected to be normalized """
        next_obs = next_obs[..., -1:]
        reward_int = self._intrinsic_reward(next_obs).numpy()
        self.update_int_reward_rms(reward_int)
        reward_int = self.normalize_int_reward(reward_int)
        return reward_int

    @tf.function
    def _intrinsic_reward(self, next_obs):
        target_feat = self.target(next_obs)
        pred_feat = self.predictor(next_obs)
        int_reward = tf.reduce_mean(tf.square(target_feat - pred_feat), axis=-1)
        return int_reward

    def update_obs_rms(self, obs):
        if obs.dtype == np.uint8 and obs.shape[-1] > 1:
            # for stacked frames, we only use
            # the most recent one for rms update
            obs = obs[..., -1:]
        self._obs_rms.update(obs)

    def update_int_reward_rms(self, reward):
        # TODO: normalize reward using the return stats
        if self._normalize_int_reward:
            self._int_reward_rms.update(reward)
    
    def update_ext_reward_rms(self, reward):
        if self._normalize_ext_reward:
            self._ext_reward_rms.update(reward)

    def normalize_obs(self, obs):
        obs = self._obs_rms.normalize(obs)
        return obs

    def normalize_int_reward(self, reward):
        if self._normalize_int_reward:
            return self._int_reward_rms.normalize(reward, subtract_mean=False)
        else:
            return reward

    def normalize_ext_reward(self, reward):
        if self._normalize_ext_reward:
            return self._ext_reward_rms.normalize(reward, subtract_mean=False)
        else:
            return reward

    def restore(self):
        import os
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._int_reward_rms, self._ext_reward_rms = \
                    cloudpickle.load(f)
        super().restore()

    def save(self, print_terminal_info=False):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump(
                (self._obs_rms, self._int_reward_rms, self._ext_reward_rms), f)
        super().save(print_terminal_info=print_terminal_info)


class Agent(RNDBase):
    @agent_config
    def __init__(self, buffer, env):
        super().__init__(buffer=buffer, env=env)

        self._n_envs = env.n_envs

        # optimizer
        self._ac_opt = Optimizer(
            self._optimizer, self.ac, self._ac_lr, 
            clip_norm=self._clip_norm)
        self._pred_opt = Optimizer(
            self._optimizer, self.predictor, self._pred_lr
        )
        # Explicitly instantiate tf.function to avoid unintended retracing
        norm_obs_shape = env.obs_shape[:-1] + (1,)
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            norm_obs=(norm_obs_shape, tf.float32, 'norm_obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            traj_ret_int=((), tf.float32, 'traj_ret_int'),
            traj_ret_ext=((), tf.float32, 'traj_ret_ext'),
            value_int=((), tf.float32, 'value_int'),
            value_ext=((), tf.float32, 'value_ext'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    def reset_states(self, states=None):
        pass

    def get_states(self):
        return None

    def __call__(self, obs, deterministic=False, **kwargs):
        obs = np.array(obs, copy=False)
        if deterministic:
            return self.action(obs, deterministic).numpy()
        else:
            out = self.action(obs, deterministic)
            action, terms = tf.nest.map_structure(lambda x: x.numpy(), out)
            terms['obs'] = obs  # return normalized obs
            return action, terms

    @tf.function()
    def action(self, obs, deterministic=False):
        if deterministic:
            act_dist = self.ac(obs, return_value=False)
            return act_dist.mode()
        else:
            act_dist, value_int, value_ext = self.ac(obs, return_value=True)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            return action, dict(logpi=logpi, value_int=value_int, value_ext=value_ext)

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            for j in range(self.N_MBS):
                data = self.buffer.sample()
                value_int = data['value_int']
                value_ext = data['value_ext']
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.learn(**data)

                terms = {k: v.numpy() for k, v in terms.items()}
                terms['value_int'] = np.mean(value_int)
                terms['max_value_int'] = np.max(value_int)
                terms['value_ext'] = np.mean(value_ext)
                terms['max_value_ext'] = np.max(value_ext)
                approx_kl = terms['approx_kl']
                del terms['approx_kl']

                self.store(**terms)
                if getattr(self, '_max_kl', 0) > 0 and approx_kl > self._max_kl:
                    break
            if getattr(self, '_max_kl', 0) > 0 and approx_kl > self._max_kl:
                pwc(f'{self._model_name}: Eearly stopping after '
                    f'{i*self.N_MBS+j+1} update(s) due to reaching max kl.',
                    f'Current kl={approx_kl:.3g}', color='blue')
                break
        self.store(approx_kl=approx_kl)
        return i * self.N_MBS + j + 1

    @tf.function
    def _learn(self, obs, norm_obs, action, traj_ret_int, traj_ret_ext, 
            value_int, value_ext, advantage, logpi):
        old_value_int, old_value_ext = value_int, value_ext
        norm_obs = tf.reshape(norm_obs, 
            (self._n_envs, self.N_STEPS // self.N_MBS, *norm_obs.shape[-3:]))
        with tf.GradientTape() as pred_tape:
            target_feat = self.target(norm_obs)
            pred_feat = self.predictor(norm_obs)
            pred_loss = tf.reduce_mean(tf.square(target_feat - pred_feat), axis=-1)
            mask = tf.random.uniform(pred_loss.shape, maxval=1., dtype=pred_loss.dtype)
            mask = tf.cast(mask < self._pred_frac, pred_loss.dtype)
            pred_loss = tf.reduce_sum(mask * pred_loss) / tf.maximum(tf.reduce_sum(mask), 1)

        with tf.GradientTape() as ac_tape:
            act_dist, value_int, value_ext = self.ac(obs, return_value=True)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            ppo_loss, entropy, approx_kl, p_clip_frac = compute_ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            v_loss_int = .5 * tf.reduce_mean(tf.square(traj_ret_int - value_int))
            v_loss_ext, v_clip_frac_ext = compute_value_loss(
                value_ext, traj_ret_ext, old_value_ext, self._clip_range)

            policy_loss = ppo_loss - self._entropy_coef * entropy
            v_loss_int = self._v_coef * v_loss_int
            v_loss_ext = self._v_coef * v_loss_ext
            ac_loss = policy_loss + v_loss_int + v_loss_ext

        terms = dict(
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            approx_kl=approx_kl, 
            p_clip_frac=p_clip_frac,
            v_clip_frac_ext=v_clip_frac_ext,
            ppo_loss=ppo_loss,
            v_loss_int=v_loss_int,
            v_loss_ext=v_loss_ext
        )
        terms['pred_norm'] = self._pred_opt(pred_tape, pred_loss)
        terms['ac_norm'] = self._ac_opt(ac_tape, ac_loss)

        return terms
