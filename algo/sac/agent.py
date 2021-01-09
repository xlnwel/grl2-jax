import itertools
import tensorflow as tf

from utility.rl_loss import n_step_target
from utility.schedule import TFPiecewiseSchedule
from utility.timer import TBTimer
from core.optimizer import Optimizer
from core.decorator import override, step_track
from algo.dqn.base import DQNBase, get_data_format, collect


class Agent(DQNBase):
    @override(DQNBase)
    def _construct_optimizers(self):
        if self._schedule_lr:
            assert isinstance(self._actor_lr, list), self._actor_lr
            assert isinstance(self._q_lr, list), self._q_lr
            self._actor_lr = TFPiecewiseSchedule(self._actor_lr)
            self._q_lr = TFPiecewiseSchedule(self._q_lr)

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._q_opt = Optimizer(self._optimizer, [self.q, self.q2], self._q_lr)

        if self.temperature.is_trainable():
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            with TBTimer('sample', 2500):
                data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            with TBTimer('learn', 2500):
                terms = self.learn(**data)
            
            self._update_target_nets()

            terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
            if self._is_per:
                self.dataset.update_priorities(terms['train/priority'], idxes)

            self.store(**terms)

        if self._to_summary(step):
            self._summary(data, terms)
        
        return self.N_UPDATES

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        next_action, next_logpi, _ = self.actor.train_step(next_obs)
        next_q_with_actor = self.target_q(next_obs, next_action)
        next_q2_with_actor = self.target_q2(next_obs, next_action)
        next_q_with_actor = tf.minimum(next_q_with_actor, next_q2_with_actor)
        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state-action':
            _, temp = self.temperature(next_obs, next_action)
        else:
            _, temp = self.temperature()
        next_value = next_q_with_actor - temp * next_logpi
        target_q = n_step_target(reward, next_value, discount, self._gamma, steps)

        terms = {}
        with tf.GradientTape() as tape:
            q = self.q(obs, action)
            q2 = self.q2(obs, action)
            q_error = target_q - q
            q2_error = target_q - q2
            q_loss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
            q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
            q_loss = q_loss + q2_loss
        terms['q_norm'] = self._q_opt(tape, q_loss)

        with tf.GradientTape() as actor_tape:
            action, logpi, actor_terms = self.actor.train_step(obs)
            terms.update(actor_terms)
            q_with_actor = self.q(obs, action)
            q2_with_actor = self.q2(obs, action)
            q_with_actor = tf.minimum(q_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(IS_ratio * 
                (temp * logpi - q_with_actor))
        self._actor_opt(actor_tape, actor_loss)

        if self.temperature.is_trainable():
            target_entropy = getattr(self, '_target_entropy', -self._action_dim)
            with tf.GradientTape() as temp_tape:
                log_temp, temp = self.temperature(obs, action)
                temp_loss = -tf.reduce_mean(IS_ratio * log_temp 
                    * tf.stop_gradient(logpi + target_entropy))
            self._temp_opt(temp_tape, temp_loss)
            terms.update(dict(
                temp=temp,
                temp_loss=temp_loss,
            ))

        if self._is_per:
            priority = self._compute_priority((tf.abs(q_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority
            
        terms.update(dict(
            actor_loss=actor_loss,
            q=q, 
            q2=q2,
            logpi=logpi,
            target_q=target_q,
            q_loss=q_loss, 
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        tf.debugging.assert_greater(priority, 0.)
        return priority

    @tf.function
    def _update_target_nets(self):
        tns = self.get_target_nets()
        ons = self.get_online_nets()
        tvars = list(itertools.chain(*[v.variables for v in tns]))
        ovars = list(itertools.chain(*[v.variables for v in ons]))
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * ovar) 
            for tvar, ovar in zip(tvars, ovars)]
