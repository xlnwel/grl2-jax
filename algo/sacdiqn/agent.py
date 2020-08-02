import numpy as np
import tensorflow as tf

from utility.rl_utils import n_step_target, huber_loss, quantile_regression_loss
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from algo.dqn.base import get_data_format, DQNBase


class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule(
                [(4e6, self._actor_lr), (7e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule(
                [(4e6, self._q_lr), (7e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._q_opt = Optimizer(self._optimizer, [self.encoder, self.q, self.q2], self._q_lr)

        if isinstance(self.temperature, float):
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

    @tf.function
    def summary(self, data, terms):
        tf.summary.histogram('entropy', terms['entropy'], step=self._env_step)
        tf.summary.histogram('next_act_probs', terms['next_act_probs'], step=self._env_step)
        tf.summary.histogram('next_act_logps', terms['next_act_logps'], step=self._env_step)
        tf.summary.histogram('next_logps', terms['next_logps'], step=self._env_step)
        tf.summary.histogram('next_qtv', terms['next_qtv'], step=self._env_step)
        tf.summary.histogram('reward', data['reward'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        if not hasattr(self, '_target_entropy'):
            # Entropy of a uniform distribution
            self._target_entropy = np.log(self._action_dim)
            self._target_entropy *= self._target_entropy_coef
        # compute target returns
        next_x = self.encoder(next_obs)
        next_act_probs, next_act_logps = self.actor.train_step(next_x)
        next_x = self.target_encoder(next_obs)
        next_act_probs = tf.expand_dims(next_act_probs, axis=1)  # [B, 1, A]
        next_act_logps = tf.expand_dims(next_act_logps, axis=1)  # [B, 1, A]
        _, next_qtv1 = self.target_q(next_x, self.N_PRIME)
        _, next_qtv2 = self.target_q2(next_x, self.N_PRIME)
        next_qtv = (next_qtv1 + next_qtv2) / 2
        tf.debugging.assert_shapes([
            [next_act_probs, (None, 1, self._action_dim)],
            [next_act_logps, (None, 1, self._action_dim)],
            [next_qtv1, (None, self.N_PRIME, self._action_dim)],
            [next_qtv2, (None, self.N_PRIME, self._action_dim)],
            [next_qtv, (None, self.N_PRIME, self._action_dim)],
        ])
        if isinstance(self.temperature, (tf.Variable)):
            temp = self.temperature
        else:
            _, temp = self.temperature()
        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        next_state_qtv = tf.reduce_sum(next_act_probs 
            * (next_qtv - temp * next_act_logps), axis=-1)
        returns = n_step_target(reward, next_state_qtv, discount, self._gamma, steps, self._tbo)
        returns = tf.expand_dims(returns, axis=1)      # [B, 1, N']

        terms['next_qtv'] = tf.reduce_sum(next_act_probs * next_qtv, axis=-1)
        terms['next_logps'] = tf.reduce_sum(next_act_probs * temp * next_act_logps, axis=-1)
        tf.debugging.assert_shapes([
            [next_state_qtv, (None, self.N_PRIME)],
            [next_act_probs, (None, 1, self._action_dim)],
            [next_act_logps, (None, 1, self._action_dim)],
            [next_qtv, (None, self.N_PRIME, self._action_dim)],
            [returns, (None, 1, self.N_PRIME)],
        ])
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            action = tf.expand_dims(action, axis=1)

            qs1, error1, qr_loss1 = self._compute_qr_loss(self.q, x, action, returns, IS_ratio)
            qs2, error2, qr_loss2 = self._compute_qr_loss(self.q2, x, action, returns, IS_ratio)

            qr_loss = qr_loss1 + qr_loss2
        terms['q_norm'] = self._q_opt(tape, qr_loss)

        with tf.GradientTape() as tape:
            act_probs, act_logps = self.actor.train_step(x)
            tf.debugging.assert_shapes([[act_probs, (None, self._action_dim)]])
            tf.debugging.assert_shapes([[act_logps, (None, self._action_dim)]])
            tf.debugging.assert_shapes([[qs1, (None, self._action_dim)]])
            tf.debugging.assert_shapes([[qs2, (None, self._action_dim)]])
            qs = (qs1 + qs2) / 2
            q = tf.reduce_sum(act_probs * qs, axis=-1)
            entropy = - tf.reduce_sum(act_probs * act_logps, axis=-1)
            actor_loss = -(q + temp * entropy)
            tf.debugging.assert_shapes([[actor_loss, (None, )]])
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        if not isinstance(self.temperature, (float, tf.Variable)):
            with tf.GradientTape() as tape:
                log_temp, temp = self.temperature()
                temp_loss = -log_temp * (self._target_entropy - entropy)
                tf.debugging.assert_shapes([[temp_loss, (None, )]])
                temp_loss = tf.reduce_mean(IS_ratio * temp_loss)
            terms['target_entropy'] = self._target_entropy
            terms['entropy_diff'] = self._target_entropy - entropy
            terms['log_temp'] = log_temp
            terms['temp'] = temp
            terms['temp_loss'] = temp_loss
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)

        if self._is_per:
            error1 = tf.reduce_max(tf.reduce_mean(error1, axis=-1), axis=-1)
            error2 = tf.reduce_max(tf.reduce_mean(error2, axis=-1), axis=-1)
            priority = self._compute_priority((tf.abs(error1) + tf.abs(error2)) / 2.)
            terms['priority'] = priority
            
        target_q = tf.reduce_mean(returns, axis=-1)
        target_q = tf.squeeze(target_q)
        terms.update(dict(
            act_probs=act_probs,
            actor_loss=actor_loss,
            q=q,
            next_value=tf.reduce_mean(next_state_qtv, axis=-1),
            logpi=act_logps,
            entropy=entropy,
            next_act_probs=next_act_probs,
            next_act_logps=next_act_logps,
            returns=returns,
            qr1_loss=qr_loss1, 
            qr2_loss=qr_loss2,
            qr_loss=qr_loss, 
            explained_variance1=explained_variance(target_q, q),
        ))

        return terms

    def _compute_qr_loss(self, q, x, action, returns, IS_ratio):
        tau_hat, qtvs, qs = q(x, self.N, return_q=True)
        qtv = tf.reduce_sum(qtvs * action, axis=-1, keepdims=True)  # [B, N, 1]
        error, qr_loss = quantile_regression_loss(qtv, returns, tau_hat, kappa=self.KAPPA, return_error=True)
            
        tf.debugging.assert_shapes([
            [qtvs, (None, self.N, self._action_dim)],
            [action, (None, 1, self._action_dim)],
            [qtv, (None, self.N, 1)],
            [qs, (None, self._action_dim)],
            [qr_loss, (None)],
        ])

        qr_loss = tf.reduce_mean(IS_ratio * qr_loss)

        return qs, error, qr_loss

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_encoder.variables + self.target_q.variables + self.target_q2.variables
        mvars = self.encoder.variables + self.q.variables + self.q2.variables
        # tvars = self.target_q.variables + self.target_q2.variables
        # mvars = self.q.variables + self.q2.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]
