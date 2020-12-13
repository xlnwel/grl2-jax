import numpy as np
import tensorflow as tf

from utility.rl_utils import n_step_target, quantile_regression_loss
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from algo.dqn.base import get_data_format, DQNBase


class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule([(4e6, self._actor_lr), (7e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule([(4e6, self._q_lr), (7e6, 1e-5)])

        actor_models = [self.actor_encoder, self.actor]
        self._actor_opt = Optimizer(self._optimizer, actor_models, self._actor_lr)
        value_models = [self.critic_encoder, self.q]
        self._value_opt = Optimizer(self._optimizer, value_models, self._q_lr)

        if self.temperature.is_trainable():
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)
            if isinstance(self._target_entropy_coef, (list, tuple)):
                self._target_entropy_coef = TFPiecewiseSchedule(self._target_entropy_coef)

    # @tf.function
    # def summary(self, data, terms):
    #     tf.summary.histogram('learn/entropy', terms['entropy'], step=self._env_step)
    #     tf.summary.histogram('learn/reward', data['reward'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        if self.temperature.is_trainable():
            # Entropy of a uniform distribution
            self._target_entropy = np.log(self._action_dim)
            target_entropy_coef = self._target_entropy_coef \
                if isinstance(self._target_entropy_coef, float) \
                else self._target_entropy_coef(self._train_step)
            target_entropy = self._target_entropy * target_entropy_coef

        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state-action':
            raise NotImplementedError
        else:
            _, temp = self.temperature()

        # compute q_target
        next_x = self.target_actor_encoder(next_obs, training=False)
        next_act_probs, next_act_logps = self.target_actor.train_step(next_x)
        next_act_probs_ext = tf.expand_dims(next_act_probs, axis=1)  # [B, 1, A]
        next_act_logps_ext = tf.expand_dims(next_act_logps, axis=1)  # [B, 1, A]
        next_x = self.target_critic_encoder(next_obs, training=False)
        _, qt_embed = self.target_quantile(next_x, self.N_PRIME)
        next_x_ext = tf.expand_dims(next_x, axis=1)
        next_qtv = self.target_q(next_x_ext, qt_embed)
        next_qtv_v = tf.reduce_sum(next_act_probs_ext 
            * (next_qtv - temp * next_act_logps_ext), axis=-1)

        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        q_target = n_step_target(reward, next_qtv_v, discount, self._gamma, steps)
        q_target = tf.expand_dims(q_target, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv_v, (None, self.N_PRIME)],
            [next_act_probs_ext, (None, 1, self._action_dim)],
            [next_act_logps_ext, (None, 1, self._action_dim)],
            [q_target, (None, 1, self.N_PRIME)],
        ])

        with tf.GradientTape() as tape:
            x = self.critic_encoder(obs, training=True)
            tau_hat, qt_embed = self.quantile(x, self.N)
            x_ext = tf.expand_dims(x, axis=1)
            action_ext = tf.expand_dims(action, axis=1)
            # q loss
            qtvs, qs = self.q(x_ext, qt_embed, return_value=True)
            qtv = tf.reduce_sum(qtvs * action_ext, axis=-1, keepdims=True)  # [B, N, 1]
            error, qr_loss = quantile_regression_loss(
                qtv, q_target, tau_hat, kappa=self.KAPPA, return_error=True)
            qr_loss = tf.reduce_mean(IS_ratio * qr_loss)

        terms['value_norm'] = self._value_opt(tape, qr_loss)

        with tf.GradientTape() as tape:
            actor_x = self.actor_encoder(obs, training=True)
            act_probs, act_logps = self.actor.train_step(actor_x)
            v = tf.reduce_sum(act_probs * (qs - temp * act_logps), axis=-1, keepdims=True)
            a = tf.reduce_sum(act_probs * tf.stop_gradient(qs - v), axis=-1)
            entropy = - tf.reduce_sum(act_probs * act_logps, axis=-1)
            actor_loss = -(a + temp * entropy)
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
            tf.debugging.assert_shapes([
                [qs, (None, self._action_dim)],
                [actor_loss, (None, )]
            ])
            fm_loss = tf.reduce_mean(IS_ratio * (.5 * tf.reduce_mean((actor_x - x)**2, axis=-1)))
            actor_total_loss = actor_loss + fm_loss
        terms['actor_norm'] = self._actor_opt(tape, actor_total_loss)

        act_probs = tf.reduce_mean(act_probs, 0)
        self.actor.update_prior(act_probs, self._prior_lr)
        if self.temperature.is_trainable():
            with tf.GradientTape() as tape:
                log_temp, temp = self.temperature(x, action)
                entropy_diff = target_entropy - entropy
                temp_loss = -log_temp * entropy_diff
                tf.debugging.assert_shapes([[temp_loss, (None, )]])
                temp_loss = tf.reduce_mean(temp_loss)
            terms['target_entropy'] = target_entropy
            terms['entropy_diff'] = entropy_diff
            terms['log_temp'] = log_temp
            terms['temp_loss'] = temp_loss
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)

        if self._is_per:
            error = tf.abs(error)
            error = tf.reduce_max(tf.reduce_mean(error, axis=-1), axis=-1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
            
        q = tf.reduce_sum(act_probs * qs, axis=-1)
        q_target = tf.reduce_mean(q_target, axis=(1, 2))
        terms.update(dict(
            steps=steps,
            reward_min=tf.reduce_min(reward),
            x=tf.reduce_mean(x),
            actor_x=tf.reduce_mean(actor_x),
            fm_loss=fm_loss,
            prob_max=tf.reduce_max(act_probs),
            prob_min=tf.reduce_min(act_probs),
            actor_loss=actor_loss,
            actor_total_loss=actor_total_loss,
            a=a,
            a_max=tf.reduce_max(a),
            a_min=tf.reduce_min(a),
            a_std=tf.math.reduce_std(a),
            q=q,
            q_max=tf.reduce_max(q),
            q_min=tf.reduce_min(q),
            q_std=tf.math.reduce_std(q),
            v=v,
            v_max=tf.reduce_max(v),
            v_min=tf.reduce_min(v),
            v_std=tf.math.reduce_std(v),
            entropy=entropy,
            entropy_max=tf.reduce_max(entropy),
            entropy_min=tf.reduce_min(entropy),
            qr_loss=qr_loss, 
            temp=temp,
            explained_variance_q=explained_variance(q_target, q),
        ))
        # for i in range(self.actor.action_dim):
        #     terms[f'prior_{i}'] = self.actor.prior[i]

        return terms

    def _compute_qr_loss(self, q, x, embed, tau_hat, returns, action, IS_ratio):
        qtvs, value = q(x, embed, return_value=True)
        if action:
            assert qtvs.shape[-1] == self._action_dim, qtvs.shape
            qtv = tf.reduce_sum(qtvs * action, axis=-1, keepdims=True)  # [B, N, 1]
        else:
            qtv = qtvs
        error, qr_loss = quantile_regression_loss(
            qtv, returns, tau_hat, kappa=self.KAPPA, return_error=True)
            
        tf.debugging.assert_shapes([
            [qtv, (None, self.N, 1)],
            [qr_loss, (None)],
        ])

        qr_loss = tf.reduce_mean(IS_ratio * qr_loss)

        return value, tf.abs(error), qr_loss

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_critic_encoder.variables + self.target_quantile.variables \
            + self.target_q.variables + self.target_actor_encoder.variables + self.target_actor.variables
        mvars = self.critic_encoder.variables + self.quantile.variables \
            + self.q.variables + self.actor_encoder.variables + self.actor.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]
