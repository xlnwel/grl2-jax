import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target
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
        self._q_opt = Optimizer(self._optimizer, [self.cnn, self.q1, self.q2], self._q_lr)

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
        tf.summary.histogram('next_qs', terms['next_qs'], step=self._env_step)
        tf.summary.histogram('reward', data['reward'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        if not hasattr(self, '_target_entropy'):
            # Entropy of a uniform distribution
            self._target_entropy = np.log(self._action_dim)
            self._target_entropy *= self._target_entropy_coef
        next_x = self.cnn(next_obs)
        next_act_probs, next_act_logps = self.actor.train_step(next_x)
        next_x = self.target_cnn(next_obs)
        next_qs1 = self.target_q1(next_x)
        next_qs2 = self.target_q2(next_x)
        next_qs = tf.minimum(next_qs1, next_qs2)
        if isinstance(self.temperature, (tf.Variable)):
            temp = self.temperature
        else:
            _, temp = self.temperature()
        terms['next_qs'] = tf.reduce_sum(next_act_probs * next_qs, axis=-1)
        terms['next_logps'] = tf.reduce_sum(next_act_probs * temp * next_act_logps, axis=-1)
        next_value = tf.reduce_sum(next_act_probs 
            * (next_qs - temp * next_act_logps), axis=-1)
        target_q = n_step_target(reward, next_value, discount, self._gamma, steps)
        target_q = tf.stop_gradient(target_q)
        tf.debugging.assert_shapes([
            [next_value, (None,)],
            [reward, (None,)],
            [next_act_probs, (None, self._action_dim)],
            [next_act_logps, (None, self._action_dim)],
            [next_qs, (None, self._action_dim)],
            [target_q, (None)],
        ])
        with tf.GradientTape() as tape:
            x = self.cnn(obs)
            qs1 = self.q1(x)
            qs2 = self.q2(tf.stop_gradient(x))
            q1 = tf.reduce_sum(qs1 * action, axis=-1)
            q2 = tf.reduce_sum(qs2 * action, axis=-1)
            qr_error1 = target_q - q1
            qr_error2 = target_q - q2
            tf.debugging.assert_shapes([
                [qs1, (None, self._action_dim)],
                [qs2, (None, self._action_dim)],
                [action, (None, self._action_dim)],
                [q1, (None, )],
                [qr_error1, (None, )],
            ])
            qr_loss1 = .5 * tf.reduce_mean(IS_ratio * qr_error1**2)
            qr_loss2 = .5 * tf.reduce_mean(IS_ratio * qr_error2**2)
            qr_loss = qr_loss1 + qr_loss2
        terms['q_norm'] = self._q_opt(tape, qr_loss)

        with tf.GradientTape() as tape:
            act_probs, act_logps = self.actor.train_step(x)
            qs = tf.minimum(qs1, qs2)
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
            priority = self._compute_priority((tf.abs(qr_error1) + tf.abs(qr_error2)) / 2.)
            terms['priority'] = priority
            
        terms.update(dict(
            act_probs=act_probs,
            actor_loss=actor_loss,
            q1=q1, 
            q2=q2,
            next_value=next_value,
            logpi=act_logps,
            entropy=entropy,
            next_act_probs=next_act_probs,
            next_act_logps=next_act_logps,
            target_q=target_q,
            qr_loss1=qr_loss1, 
            qr_loss2=qr_loss2,
            qr_loss=qr_loss, 
        ))

        return terms

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_cnn.variables + self.target_q1.variables + self.target_q2.variables
        mvars = self.cnn.variables + self.q1.variables + self.q2.variables
        # tvars = self.target_q1.variables + self.target_q2.variables
        # mvars = self.q1.variables + self.q2.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]
