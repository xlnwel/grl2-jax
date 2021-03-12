import functools
import numpy as np
import tensorflow as tf

from utility.rl_loss import n_step_target
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from core.decorator import override
from algo.dqn.base import DQNBase, get_data_format, collect


class Agent(DQNBase):
    @override(DQNBase)
    def _construct_optimizers(self):
        if self._schedule_lr:
            assert isinstance(self._actor_lr, list), self._actor_lr
            assert isinstance(self._value_lr, list), self._value_lr
            self._actor_lr = TFPiecewiseSchedule(self._actor_lr)
            self._value_lr = TFPiecewiseSchedule(self._value_lr)
        
        PartialOpt = functools.partial(
            Optimizer,
            name=self._optimizer,
            weight_decay=getattr(self, '_weight_decay', None),
            clip_norm=getattr(self, '_clip_norm', None),
            epsilon=getattr(self, '_epsilon', 1e-7)
        )
        self._actor_opt = PartialOpt(models=self.actor, lr=self._actor_lr)
        value_models = [self.encoder, self.q]
        self._value_opt = PartialOpt(models=value_models, lr=self._value_lr)

        if self.temperature.is_trainable():
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

    # @tf.function
    # def summary(self, data, terms):
    #     tf.summary.histogram('learn/entropy', terms['entropy'], step=self._env_step)
    #     tf.summary.histogram('learn/reward', data['reward'], step=self._env_step)

    @override(DQNBase)
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state-action':
            raise NotImplementedError
        else:
            _, temp = self.temperature()

        next_x = self.target_encoder(next_obs, training=False)
        next_act_probs, next_act_logps = self.target_actor.train_step(next_x)
        # next_x = self.target_encoder(next_obs)
        next_qs = self.target_q(next_x)
        if self._soft_target:
            next_value = tf.reduce_sum(next_act_probs 
                * (next_qs - temp * next_act_logps), axis=-1)
        else:
            next_value = tf.reduce_sum(next_act_probs * next_qs, axis=-1)
        q_target = n_step_target(reward, next_value, discount, self._gamma, steps)
        tf.debugging.assert_shapes([
            [next_value, (None,)],
            [reward, (None,)],
            [next_act_probs, (None, self._action_dim)],
            [next_act_logps, (None, self._action_dim)],
            [next_qs, (None, self._action_dim)],
            [q_target, (None)],
        ])
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            qs, error, value_losss = self._compute_value_losss(
                self.q, x, action, q_target, IS_ratio)
        terms['value_norm'] = self._value_opt(tape, value_losss)

        with tf.GradientTape() as tape:
            act_probs, act_logps = self.actor.train_step(x)
            q = tf.reduce_sum(act_probs * qs, axis=-1)
            entropy = - tf.reduce_sum(act_probs * act_logps, axis=-1)
            actor_loss = -(q + temp * entropy)
            tf.debugging.assert_shapes([[actor_loss, (None, )]])
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        if self.temperature.is_trainable():
            # Entropy of a uniform distribution
            self._target_entropy = np.log(self._action_dim)
            target_entropy_coef = self._target_entropy_coef \
                if isinstance(self._target_entropy_coef, float) \
                else self._target_entropy_coef(self._train_step)
            target_entropy = self._target_entropy * target_entropy_coef
            with tf.GradientTape() as tape:
                log_temp, temp = self.temperature()
                entropy_diff = entropy - target_entropy
                temp_loss = log_temp * entropy_diff
                tf.debugging.assert_shapes([[temp_loss, (None, )]])
                temp_loss = tf.reduce_mean(IS_ratio * temp_loss)
            terms['target_entropy'] = target_entropy
            terms['entropy_diff'] = entropy_diff
            terms['log_temp'] = log_temp
            terms['temp_loss'] = temp_loss
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)

        if self._is_per:
            priority = self._compute_priority(error)
            terms['priority'] = priority
            
        terms.update(dict(
            steps=steps,
            reward_min=tf.reduce_min(reward),
            actor_loss=actor_loss,
            q=q, 
            entropy=entropy,
            entropy_max=tf.reduce_max(entropy),
            entropy_min=tf.reduce_min(entropy),
            value_losss=value_losss, 
            temp=temp,
            explained_variance_q=explained_variance(q_target, q),
        ))

        return terms

    def _compute_value_losss(self, q_fn, x, action, returns, IS_ratio):
        qs = q_fn(x)
        q = tf.reduce_sum(qs * action, axis=-1)
        q_error = returns - q
        value_losss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
        return qs, tf.abs(q_error), value_losss
