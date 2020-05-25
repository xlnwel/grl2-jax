import cloudpickle
import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class PPOBase(BaseAgent):
    def __init__(self, buffer, env):
        from env.wrappers import get_wrapper_by_name
        from utility.utils import RunningMeanStd
        self.buffer = buffer
        axis = None if get_wrapper_by_name(env, 'Env') else 0
        self._obs_rms = RunningMeanStd(axis=axis)
        self._reward_rms = RunningMeanStd(axis=axis)
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

    def normalize_reward(self, reward):
        if self._normalize_reward:
            self._reward_rms.update(reward)
            return self._reward_rms.normalize(reward, subtract_mean=False)
        else:
            return reward
    
    def normalize_obs(self, obs, update_rms=False):
        if self._normalize_obs:
            if update_rms:
                self._obs_rms.update(obs)
            return self._obs_rms.normalize(obs)
        else:
            return obs

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

class Agent(PPOBase):
    @agent_config
    def __init__(self, buffer, env):
        super().__init__(buffer, env=env)

        # optimizer
        if getattr(self, 'schedule_lr', False):
            self._lr = TFPiecewiseSchedule(
                [(300, self._lr), (1000, 5e-5)])
        self._optimizer = Optimizer(
            self._optimizer, self.ac, self._lr, 
            clip_norm=self._clip_norm)
        self._ckpt_models['optimizer'] = self._optimizer

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, self._dtype, 'obs'),
            action=(env.action_shape, self._dtype, 'action'),
            traj_ret=((), self._dtype, 'traj_ret'),
            value=((), self._dtype, 'value'),
            advantage=((), self._dtype, 'advantage'),
            logpi=((), self._dtype, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    def reset_states(self, states=None):
        pass

    def get_states(self):
        return None

    def __call__(self, obs, deterministic=False, update_rms=False, **kwargs):
        obs = np.array(obs, copy=False)
        obs = self.normalize_obs(obs, update_rms)
        if deterministic:
            return self.action(obs, deterministic).numpy()
        else:
            out = self.action(obs, deterministic)
            action, terms = tf.nest.map_structure(lambda x: x.numpy(), out)
            terms['obs'] = obs
            return action, terms

    @tf.function
    def action(self, obs, deterministic=False):
        # if obs.dtype == np.uint8:
        #     obs = tf.cast(obs, self._dtype) / 255.
        if deterministic:
            act_dist = self.ac(obs, return_value=False)
            return act_dist.mode()
        else:
            act_dist, value = self.ac(obs, return_value=True)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            return action, dict(logpi=logpi, value=value)

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            for j in range(self.N_MBS):
                data = self.buffer.sample()
                if data['obs'].dtype == np.uint8:
                    data['obs'] = data['obs'] / 255.
                value = data['value']
                data = {k: tf.convert_to_tensor(v, self._dtype) for k, v in data.items()}
                terms = self.learn(**data)

                terms = {k: v.numpy() for k, v in terms.items()}

                terms['value'] = np.mean(value)
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
        if not isinstance(self._lr, float):
            step = tf.cast(self._env_steps, self._dtype)
            self.store(lr=self._lr(step))
        return i * self.N_MBS + j + 1

    @tf.function
    def _learn(self, obs, action, traj_ret, value, advantage, logpi):
        old_value = value
        with tf.GradientTape() as tape:
            act_dist, value = self.ac(obs, return_value=True)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            ppo_loss, entropy, approx_kl, p_clip_frac = compute_ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self._clip_range)

            with tf.name_scope('total_loss'):
                policy_loss = (ppo_loss - self._entropy_coef * entropy)
                value_loss = self._value_coef * value_loss
                total_loss = policy_loss + value_loss

        terms = dict(
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            approx_kl=approx_kl, 
            p_clip_frac=p_clip_frac,
            v_clip_frac=v_clip_frac,
            ppo_loss=ppo_loss,
            value_loss=value_loss
        )
        terms['grads_norm'] = self._optimizer(tape, total_loss)

        return terms
