import tensorflow as tf

from utility.rl_utils import n_step_target
from utility.schedule import TFPiecewiseSchedule
from core.decorator import override, step_track
from core.optimizer import Optimizer
from algo.dqn.base import DQNBase, get_data_format, collect


class Agent(DQNBase):
    @override(DQNBase)
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule([(2e5, self._actor_lr), (1e6, 1e-5)])
            self._value_lr = TFPiecewiseSchedule([(2e5, self._value_lr), (1e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._value_opt = Optimizer(self._optimizer, self.value, self._value_lr)
        self._q_opt = Optimizer(self._optimizer, [self.q, self.q2], self._value_lr)

        if self.temperature.is_trainable():
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

    def __call__(self, obs, evaluation=False, **kwargs):
        return self.model.action(
            obs, 
            evaluation=evaluation, 
            epsilon=self._act_eps).numpy()

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            with self._sample_timer:
                data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            with self._train_timer:
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
        q_value = lambda q, obs, act: q(tf.concat([obs, act], -1)).mode()
        with tf.GradientTape() as actor_tape:
            act_dist, terms = self.actor.step(obs)
            new_action = act_dist.sample(one_hot=True)
            new_logpi = act_dist.log_prob(new_action)
            if isinstance(self.temperature, (float, tf.Variable)):
                temp = self.temperature
            else:
                _, temp = self.temperature(obs)
            q_with_actor = q_value(self.q, obs, new_action)
            q2_with_actor = q_value(self.q2, obs, new_action)
            q_with_actor = tf.minimum(q_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(IS_ratio * 
                (temp * new_logpi - q_with_actor))
        
        if self.temperature.is_trainable():
            with tf.GradientTape() as temp_tape:
                log_temp, temp = self.temperature(obs)
                temp_loss = -tf.reduce_mean(IS_ratio * log_temp 
                    * tf.stop_gradient(new_logpi + self._target_entropy))
                terms['temp_loss'] = temp_loss
                terms['temp'] = temp

        with tf.GradientTape() as value_tape:
            value = self.value(obs).mode()
            target_value = q_with_actor - temp * new_logpi
            value_loss = .5 * tf.reduce_mean(IS_ratio * (target_value - value)**2)

        with tf.GradientTape() as q_tape:
            q = q_value(self.q, obs, action)
            q2 = q_value(self.q2, obs, action)
            q = tf.minimum(q, q2)
            next_value = self.target_value(next_obs).mode()
            
            target_q = n_step_target(reward, next_value, discount, self._gamma, steps)
            target_q = tf.stop_gradient(target_q)
            q_error = target_q - q
            q2_error = target_q - q2
            value_losss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
            q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
            value_losss = value_losss + q2_loss

        if self._is_per:
            priority = self._compute_priority((tf.abs(q_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority

        terms['actor_norm'] = self._actor_opt(actor_tape, actor_loss)
        terms['value_norm'] = self._value_opt(value_tape, value_loss)
        terms['value_norm'] = self._q_opt(q_tape, value_losss)
        if not isinstance(self.temperature, (float, tf.Variable)):
            terms['temp_norm'] = self._temp_opt(temp_tape, temp_loss)
            
        terms.update(dict(
            actor_loss=actor_loss,
            q=q, 
            q2=q2,
            logpi=new_logpi,
            action_entropy=act_dist.entropy(),
            target_q=target_q,
            value_losss=value_losss, 
            q2_loss=q2_loss,
        ))

        return terms
