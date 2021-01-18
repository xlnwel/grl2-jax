import tensorflow as tf

from utility.tf_utils import softmax, log_softmax
from utility.rl_loss import n_step_target, huber_loss
from algo.dqn.base import DQNBase, get_data_format, collect


class Agent(DQNBase):
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        # compute target returns
        x = self.target_encoder(obs)
        qs = self.target_q(x)
        logpi = log_softmax(qs, self._tau, axis=-1)
        logpi_a = tf.reduce_sum(logpi * action, axis=-1)
        logpi_a = tf.clip_by_value(logpi_a, self._clip_logpi_min, 0)
        reward = reward + self._alpha * logpi_a

        next_x = self.target_encoder(next_obs)
        if self._double:
            next_x_online = self.encoder(next_obs)
            next_qs = self.q(next_x_online)
        else:
            next_qs = self.target_q(next_x)
        next_pi = softmax(next_qs, self._tau)
        next_logpi = log_softmax(next_qs, self._tau)
        next_v = tf.reduce_sum((next_qs - next_logpi)*next_pi, axis=-1)
        
        returns = n_step_target(reward, next_v, discount, self._gamma, steps, self._tbo)

        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            q = self.q(x, action)
            error = returns - q
            loss = tf.reduce_mean(IS_ratio * loss_fn(error))

        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            loss=loss,
        ))

        return terms

    def reset_noisy(self):
        self.q.reset_noisy()
