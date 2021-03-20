import functools
import tensorflow as tf

from utility.rl_loss import retrace
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from core.decorator import override
from algo.mrdqn.base import RDQNBase, get_data_format, collect


class Agent(RDQNBase):
    @override(RDQNBase)
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
        value_models = [self.encoder, self.q]
        self._value_opt = PartialOpt(models=value_models, lr=self._value_lr)
        
        if 'actor' in self.model:
            self._actor_opt = PartialOpt(models=self.actor, lr=self._actor_lr)
        if self.temperature.is_trainable():
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)
            if isinstance(self._target_entropy_coef, (list, tuple)):
                self._target_entropy_coef = TFPiecewiseSchedule(self._target_entropy_coef)

    @tf.function
    def _learn(self, obs, action, reward, discount, mu, mask, 
                IS_ratio=1, state=None, additional_rnn_input=[]):
        mask = tf.expand_dims(mask, -1)
        if additional_rnn_input != []:
            prev_action, prev_reward = additional_rnn_input
            prev_action = tf.concat([prev_action, action[:, :-1]], axis=1)
            prev_reward = tf.concat([prev_reward, reward[:, :-1]], axis=1)
            add_inp = [prev_action, prev_reward]
        else:
            add_inp = additional_rnn_input
            
        target, terms = self._compute_target(
            obs, action, reward, discount, 
            mu, mask, state, add_inp)
        if self._burn_in:
            bis = self._burn_in_size
            ss = self._sample_size - bis
            bi_obs, obs, _ = tf.split(obs, [bis, ss, 1], 1)
            bi_mask, mask, _ = tf.split(mask, [bis, ss, 1], 1)
            _, mu, _ = tf.split(mu, [bis, ss, 1], 1)
            if add_inp:
                bi_add_inp, add_inp, _ = zip(*[tf.split(v, [bis, ss, 1]) for v in add_inp])
            else:
                bi_add_inp = []
            _, state = self._compute_embed(bi_obs, bi_mask, state, bi_add_inp)
        else:
            obs, _ = tf.split(obs, [self._sample_size, 1], 1)
            mask, _ = tf.split(mask, [self._sample_size, 1], 1)
            mu, _ = tf.split(mu, [self._sample_size, 1], 1)
        action, _ = tf.split(action, [self._sample_size, 1], 1)

        with tf.GradientTape() as tape:
            x, _ = self._compute_embed(obs, mask, state, add_inp)
            
            qs = self.q(x)
            q = tf.reduce_sum(qs * action, -1)
            error = target - q
            value_loss = tf.reduce_mean(.5 * error**2, axis=-1)
            value_loss = tf.reduce_mean(IS_ratio * value_loss)
        tf.debugging.assert_shapes([
            [q, (None, self._sample_size)],
            [target, (None, self._sample_size)],
            [error, (None, self._sample_size)],
            [IS_ratio, (None,)],
            [value_loss, ()]
        ])
        terms['value_norm'] = self._optimizer(tape, value_loss)
        
        if 'actor' in self.model:
            with tf.GradientTape as tape:
                pi, logpi = self.actor.train_step(x)
                loo_loss = tf.math.minimum(self._loo_c, 1 / mu) * error * logpi + tf.reduce_sum(qs * pi, axis=-1)
                loo_loss = tf.reduce_mean(loo_loss, axis=-1)
                actor_loss = tf.reduce_mean(IS_ratio, loo_loss)
            tf.debugging.assert_shape([
                [loo_loss, (None)],
            ])
            terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms.update(dict(
            q=q,
            mu_min=tf.reduce_min(mu),
            mu=mu,
            mu_std=tf.math.reduce_std(mu),
            target=target,
            value_loss=value_loss,
            ratio=tf.reduce_mean(pi/mu),
            actor_loss=actor_loss,
        ))

        return terms
    
    def _compute_target(self, obs, action, reward, discount, 
                        mu, mask, state, add_inp):
        terms = {}
        x, _ = self._compute_embed(obs, mask, state, add_inp, online=False)
        if self._burn_in:
            bis = self._burn_in_size
            ss = self._sample_size - bis
            _, reward = tf.split(reward, [bis, ss], 1)
            _, discount = tf.split(discount, [bis, ss], 1)
            _, next_mu_a = tf.split(mu, [bis+1, ss], 1)
            _, next_x = tf.split(x, [bis+1, ss], 1)
            _, next_action = tf.split(action, [bis+1, ss], 1)
        else:
            _, next_mu_a = tf.split(mu, [1, self._sample_size], 1)
            _, next_x = tf.split(x, [1, self._sample_size], 1)
            _, next_action = tf.split(action, [1, self._sample_size], 1)

        next_pi, next_logpi = self.actor.train_steps(next_x)
        next_qs = self.target_q(next_x)
        if self._probabilistic_regularization == 'entropy':
            regularization = tf.reduce_sum(next_pi * next_logpi, axis=-1)
            terms['next_entropy'] = - regularization / self._tau
        else:
            regularization = None

        discount = discount * self._gamma
        target = retrace(
            reward, next_qs, next_action, 
            next_pi, next_mu_a, discount,
            lambda_=self._lambda, 
            axis=1, tbo=self._tbo,
            regularization=regularization)

        return target, terms
