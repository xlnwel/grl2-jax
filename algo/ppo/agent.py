import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from utility.display import pwc
from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from core.optimizer import Optimizer
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class Agent(BaseAgent):
    @agent_config
    def __init__(self, env):
        self._dtype = global_policy().compute_dtype

        # optimizer
        if getattr(self, 'schedule_lr', False):
            self._learning_rate = TFPiecewiseSchedule(
                [(300, self._learning_rate), (1000, 5e-5)])

        self._optimizer = Optimizer(
            self._optimizer, self.ac, self._learning_rate, 
            clip_norm=self._clip_norm, epsilon=self._epsilon)
        self._ckpt_models['optimizer'] = self._optimizer

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, self._dtype, 'obs'),
            action=(env.action_shape, self._dtype, 'action'),
            traj_ret=((), self._dtype, 'traj_ret'),
            value=((), self._dtype, 'value'),
            advantage=((), self._dtype, 'advantage'),
            old_logpi=((), self._dtype, 'old_logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    def reset_states(self, states=None):
        return

    def __call__(self, obs, deterministic=False):
        out = self.action(obs, deterministic)
        return tf.nest.map_structure(lambda x: x.numpy(), out)

    @tf.function
    def action(self, obs, deterministic=False):
        if obs.dtype == np.uint8:
            obs = tf.cast(obs, self._dtype) / 255.
        if deterministic:
            act_dist = self.ac(obs, return_value=False)
            return act_dist.mode()
        else:
            act_dist, value = self.ac(obs, return_value=True)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            return action, logpi, value

    def learn_log(self, buffer, step):
        for i in range(self.N_UPDATES):
            for j in range(self.N_MBS):
                data = buffer.sample()
                if data['obs'].dtype == np.uint8:
                    data['obs'] /= 255.
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
                pwc(f'Eearly stopping after {i*self.N_MBS+j+1} update(s) due to reaching max kl.',
                    f'Current kl={approx_kl:.3g}', color='blue')
                break
        self.store(approx_kl=approx_kl)
        if not isinstance(self._learning_rate, float):
            step = tf.cast(self.global_steps, self._dtype)
            self.store(learning_rate=self._learning_rate(step))

    @tf.function
    def _learn(self, obs, action, traj_ret, value, advantage, old_logpi):
        old_value = value
        with tf.GradientTape() as tape:
            act_dist, value = self.ac(obs, return_value=True)
            logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            ppo_loss, entropy, approx_kl, p_clip_frac = compute_ppo_loss(
                logpi, old_logpi, advantage, self._clip_range, entropy)
            # value loss
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self._clip_range)

            with tf.name_scope('total_loss'):
                policy_loss = (ppo_loss - self._entropy_coef * entropy)
                value_loss = self._value_coef * value_loss
                total_loss = policy_loss + value_loss

        terms = dict(
            entropy=entropy, 
            approx_kl=approx_kl, 
            p_clip_frac=p_clip_frac,
            v_clip_frac=v_clip_frac,
            ppo_loss=ppo_loss,
            value_loss=value_loss
        )
        terms['grads_norm'] = self._optimizer(tape, total_loss)

        return terms
