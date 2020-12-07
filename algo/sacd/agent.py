import numpy as np
import tensorflow as tf

from utility.rl_utils import n_step_target
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from algo.dqn.base import get_data_format, DQNBase


class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule([(4e6, self._actor_lr), (7e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule([(4e6, self._q_lr), (7e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        q_models = [self.encoder, self.q]
        self._twin_q = hasattr(self, 'q2')
        if self._twin_q:
            q_models.append(self.q2)
        self._q_opt = Optimizer(self._optimizer, q_models, self._q_lr)

        if self.temperature.is_trainable():
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

    @tf.function
    def summary(self, data, terms):
        tf.summary.histogram('entropy', terms['entropy'], step=self._env_step)
        tf.summary.histogram('next_act_probs', terms['next_act_probs'], step=self._env_step)
        tf.summary.histogram('next_act_logps', terms['next_act_logps'], step=self._env_step)
        tf.summary.histogram('entropy_next', terms['entropy_next'], step=self._env_step)
        tf.summary.histogram('next_qs', terms['next_qs'], step=self._env_step)
        tf.summary.histogram('reward', data['reward'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        if not hasattr(self, '_target_entropy'):
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

        next_x = self.target_encoder(next_obs)
        next_act_probs, next_act_logps = self.actor.train_step(next_x)
        # next_x = self.target_encoder(next_obs)
        next_qs = self.target_q(next_x)
        next_value = tf.reduce_sum(next_act_probs 
            * (next_qs - temp * next_act_logps), axis=-1)
        target_q = n_step_target(reward, next_value, discount, self._gamma, steps)
        tf.debugging.assert_shapes([
            [next_value, (None,)],
            [reward, (None,)],
            [next_act_probs, (None, self._action_dim)],
            [next_act_logps, (None, self._action_dim)],
            [next_qs, (None, self._action_dim)],
            [target_q, (None)],
        ])
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            qs, q_error, q_loss = self._compute_q_loss(
                self.q, x, action, target_q, IS_ratio)
        terms['q_norm'] = self._q_opt(tape, q_loss)

        with tf.GradientTape() as tape:
            act_probs, act_logps = self.actor.train_step(x)
            q = tf.reduce_sum(act_probs * qs, axis=-1)  # recompute q to allow gradient to pass through
            entropy = - tf.reduce_sum(act_probs * act_logps, axis=-1)
            actor_loss = -(q + temp * entropy)
            tf.debugging.assert_shapes([[actor_loss, (None, )]])
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        if self.temperature.is_trainable():
            with tf.GradientTape() as tape:
                log_temp, temp = self.temperature()
                temp_loss = -log_temp * (target_entropy - entropy)
                tf.debugging.assert_shapes([[temp_loss, (None, )]])
                temp_loss = tf.reduce_mean(IS_ratio * temp_loss)
            terms['target_entropy'] = target_entropy
            terms['entropy_diff'] = target_entropy - entropy
            terms['log_temp'] = log_temp
            terms['temp'] = temp
            terms['temp_loss'] = temp_loss
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)

        if self._is_per:
            priority = self._compute_priority(q_error)
            terms['priority'] = priority
            
        terms.update(dict(
            act_probs=act_probs,
            actor_loss=actor_loss,
            q=q, 
            next_value=next_value,
            logpi=act_logps,
            entropy=entropy,
            next_act_probs=next_act_probs,
            next_act_logps=next_act_logps,
            target_q=target_q,
            q_loss=q_loss, 
            explained_variance=explained_variance(target_q, q)
        ))

        return terms

    def _compute_q_loss(self, q_fn, x, action, returns, IS_ratio):
        qs = q_fn(x)
        q = tf.reduce_sum(qs * action, axis=-1)
        q_error = returns - q
        q_loss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
        return qs, tf.abs(q_error), q_loss

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_encoder.variables + self.target_q.variables
        mvars = self.encoder.variables + self.q.variables
        if self._twin_q:
            tvars += self.target_q2.variables
            mvars += self.q2.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]
