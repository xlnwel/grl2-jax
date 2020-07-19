import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target, huber_loss, quantile_regression_loss
from utility.utils import Every
from utility.schedule import TFPiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.dqn.base import get_data_format, DQNBase


class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule(
                [(2e5, self._actor_lr), (1e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule(
                [(2e5, self._q_lr), (1e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._q_opt = Optimizer(self._optimizer, [self.cnn, self.q], self._q_lr)

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
        next_x = self.cnn(next_obs)
        next_act_probs, next_act_logps = self.actor.train_step(next_x)
        next_act_probs = tf.expand_dims(next_act_probs, axis=1)  # [B, 1, A]
        next_act_logps = tf.expand_dims(next_act_logps, axis=1)  # [B, 1, A]
        _, next_qtv = self.target_q(next_x, self.N_PRIME)
        tf.debugging.assert_shapes([
            [next_act_probs, (None, 1, self._action_dim)],
            [next_act_logps, (None, 1, self._action_dim)],
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
        
        terms['temp'] = temp
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
            x = self.cnn(obs)
            tau_hat, qtvs, qs = self.q(x, self.N, return_q=True)
            action = tf.expand_dims(action, axis=1)
            qtv = tf.reduce_sum(qtvs * action, axis=-1, keepdims=True)  # [B, N, 1]
            qr_loss = quantile_regression_loss(qtv, returns, tau_hat, kappa=self.KAPPA)
            qr_loss = tf.reduce_mean(IS_ratio * qr_loss)
        terms['q_norm'] = self._q_opt(tape, qr_loss)

        with tf.GradientTape() as tape:
            act_probs, act_logps = self.actor.train_step(x)
            tf.debugging.assert_shapes([[act_probs, (None, self._action_dim)]])
            tf.debugging.assert_shapes([[act_logps, (None, self._action_dim)]])
            tf.debugging.assert_shapes([[qs, (None, self._action_dim)]])
            q = tf.reduce_sum(act_probs * qs, axis=-1)
            entropy = - tf.reduce_sum(act_probs * act_logps, axis=-1)
            tf.debugging.assert_shapes([
                [q, (None,)],
                [entropy, (None,)],
            ])
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
            terms['temp_loss'] = temp_loss
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)

        if self._is_per:
            error = tf.reduce_max(tf.reduce_mean(tf.abs(error), axis=2), axis=1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
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
            qr_loss=qr_loss,
        ))

        return terms

    @tf.function
    def _sync_target_nets(self):
        # TODO: change this if using a target cnn
        [tv.assign(mv) for mv, tv in zip(
            self.q.variables, self.target_q.variables)]
