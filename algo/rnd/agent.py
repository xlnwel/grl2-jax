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
    def __init__(self, dataset, env):
        from env.wrappers import get_wrapper_by_name
        from utility.utils import RunningMeanStd
        self.dataset = dataset
        axis = 0 if get_wrapper_by_name(env, 'Env') else (0, 1)
        self._obs_rms = RunningMeanStd(axis=axis, clip=5)
        self._returns_int = 0
        self._int_return_rms = RunningMeanStd(axis=axis)
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

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
        # we intend to do the discoutning backwards in time as the future is yet to ocme
        self._returns_int = reward + self._gamma_int * self._returns_int
        return self._returns_int

    def update_obs_rms(self, obs):
        if obs.dtype == np.uint8 and obs.shape[-1] > 1:
            # for stacked frames, we only use
            # the most recent one for rms update
            obs = obs[..., -1:]
        if len(obs.shape) == 4:
            obs = np.expand_dims(obs, 1)
        assert len(obs.shape) == 5, obs.shape
        assert obs.dtype == np.uint8, obs.dtype
        self._obs_rms.update(obs)

    def _update_int_return_rms(self, reward):
        while len(reward.shape) < 2:
            reward = np.expand_dims(reward, 1)
        assert len(reward.shape) == 2
        self._int_return_rms.update(reward)

    def normalize_obs(self, obs):
        obs = self._obs_rms.normalize(obs[..., -1:])
        return obs

    def _normalize_int_reward(self, reward):
        return self._int_return_rms.normalize(reward, subtract_mean=False)

    def restore(self):
        import os
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._int_return_rms = \
                    cloudpickle.load(f)
            print('RMSs are restored')
        super().restore()

    def save(self, print_terminal_info=False):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump(
                (self._obs_rms, self._int_return_rms), f)
        super().save(print_terminal_info=print_terminal_info)


class Agent(RNDBase):
    @agent_config
    def __init__(self, dataset, env):
        super().__init__(dataset=dataset, env=env)

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

        import collections
        self.eval_reward_int = []
        self.eval_reward_ext = []
        self.eval_action = []

    def store_eval_reward(self, reward):
        self.eval_reward_ext.append(np.squeeze(reward))

    def retrieve_eval_rewards(self):
        reward_int = np.array(self.eval_reward_int)
        reward_ext = np.array(self.eval_reward_ext)
        self.eval_reward_int.clear()
        self.eval_reward_ext.clear()
        return reward_int, reward_ext

    def retrieve_eval_actions(self):
        action = np.array(self.eval_action)
        self.eval_action.clear()
        return action

    def reset_states(self, states=None):
        pass

    def get_states(self):
        return None

    def __call__(self, obs, deterministic=False, **kwargs):
        if deterministic:
            norm_obs = np.expand_dims(obs, 1)
            norm_obs = self.normalize_obs(norm_obs)
            reward_int = self.compute_int_reward(norm_obs)
            self.eval_reward_int.append(np.squeeze(reward_int))
            action = self.action(obs, deterministic).numpy()
            self.eval_action.append(action)
            return action
        else:
            out = self.action(obs, deterministic)
            action, terms = tf.nest.map_structure(lambda x: x.numpy(), out)
            return action, terms

    @tf.function(experimental_relax_shapes=True)
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
                data = self.dataset.sample()
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
            #     if getattr(self, '_max_kl', 0) > 0 and approx_kl > self._max_kl:
            #         break
            # if getattr(self, '_max_kl', 0) > 0 and approx_kl > self._max_kl:
            #     pwc(f'{self._model_name}: Eearly stopping after '
            #         f'{i*self.N_MBS+j+1} update(s) due to reaching max kl.',
            #         f'Current kl={approx_kl:.3g}', color='blue')
            #     break
        self.store(approx_kl=approx_kl)
        return i * self.N_MBS + j + 1

    @tf.function
    def _learn(self, obs, norm_obs, action, traj_ret_int, traj_ret_ext, 
            value_int, value_ext, advantage, logpi):
        old_value_int, old_value_ext = value_int, value_ext
        norm_obs = tf.reshape(norm_obs, 
            (self._n_envs, self.N_STEPS // self.N_MBS, *norm_obs.shape[-3:]))
        with tf.GradientTape() as pred_tape:
            target_feat = tf.stop_gradient(self.target(norm_obs))
            pred_feat = self.predictor(norm_obs)
            pred_loss = tf.reduce_mean(tf.square(target_feat - pred_feat), axis=-1)
            tf.debugging.assert_shapes([[pred_loss, (self._n_envs, self.N_STEPS // self.N_MBS)]])
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
            v_loss_ext = .5 * tf.reduce_mean(tf.square(traj_ret_ext - value_ext))
            # v_loss_ext, v_clip_frac_ext = compute_value_loss(
            #     value_ext, traj_ret_ext, old_value_ext, self._clip_range)

            entropy_loss = - self._entropy_coef * entropy
            policy_loss = ppo_loss + entropy_loss
            v_loss_int = self._v_coef * v_loss_int
            v_loss_ext = self._v_coef * v_loss_ext
            ac_loss = policy_loss + v_loss_int + v_loss_ext

        target_feat_mean, target_feat_var = tf.nn.moments(target_feat, axes=[0, 1])
        pred_feat_mean, pred_feat_var = tf.nn.moments(pred_feat, axes=[0, 1])
        terms = dict(
            target_feat_mean=target_feat_mean,
            target_feat_var=target_feat_var,
            target_feat_max=tf.reduce_max(tf.abs(target_feat)),
            pred_feat_mean=pred_feat_mean,
            pred_feat_var=pred_feat_var,
            pred_feat_max=tf.reduce_max(tf.abs(pred_feat)),
            pred_loss=pred_loss,
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            approx_kl=approx_kl, 
            p_clip_frac=p_clip_frac,
            # v_clip_frac_ext=v_clip_frac_ext,
            ppo_loss=ppo_loss,
            entropy_loss=entropy_loss,
            v_loss_int=v_loss_int,
            v_loss_ext=v_loss_ext
        )
        terms['pred_norm'] = self._pred_opt(pred_tape, pred_loss)
        terms['ac_norm'] = self._ac_opt(ac_tape, ac_loss)

        return terms
