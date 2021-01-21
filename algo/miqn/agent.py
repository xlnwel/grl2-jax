import tensorflow as tf

from utility.tf_utils import softmax, log_softmax
from utility.rl_loss import n_step_target, quantile_regression_loss
from core.decorator import override
from algo.dqn.base import DQNBase, get_data_format, collect


class Agent(DQNBase):
    @override(DQNBase)
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        # compute the target
        x = self.target_encoder(obs)
        _, qt_embed = self.target_quantile(x, self.N_PRIME)
        qtvs = self.target_q(x, qt_embed)
        logpi = log_softmax(qtvs, self._tau, axis=-1)
        logpi_a = tf.reduce_sum(logpi * action, axis=-1)
        logpi_a = tf.clip_by_value(logpi_a, self._clip_logpi_min, 0)
        reward = reward[:, None]
        reward = reward + self._alpha * logpi_a

        if self._double:
            next_x_online = self.encoder(next_obs)
            _, next_qt_embed_online = self.quantile(next_x_online, self.N)
            next_qtvs = self.q(next_x_online, next_qt_embed_online)
        else:
            next_x = self.target_encoder(next_obs)
            _, next_qt_embed = self.target_quantile(next_x, self.N_PRIME)
            next_qtvs = self.target_q(next_x, next_qt_embed)
        next_pi = softmax(next_qtvs, self._tau)
        next_logpi = log_softmax(next_qtvs, self._tau)
        next_qtv = tf.reduce_sum((next_qtvs - next_logpi)*next_pi, axis=-1)
        
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        target = n_step_target(reward, next_qtv, discount, self._gamma, steps)
        target = tf.expand_dims(target, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv, (None, self.N_PRIME)],
            [target, (None, 1, self.N_PRIME)],
        ])

        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            tau_hat, qt_embed = self.quantile(x, self.N)
            qtv = self.q(x, qt_embed, action)
            qtv = tf.expand_dims(qtv, axis=-1)  # [B, N, 1]
            error, qr_loss = quantile_regression_loss(
                qtv, target, tau_hat, kappa=self.KAPPA, return_error=True)
            loss = tf.reduce_mean(IS_ratio * qr_loss)

        terms['norm'] = self._optimizer(tape, loss)

        if self._is_per:
            error = tf.abs(error)
            error = tf.reduce_max(tf.reduce_mean(error, axis=-1), axis=-1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        terms.update(dict(
            target=target,
            q=tf.reduce_mean(qtv),
            qr_loss=qr_loss,
            loss=loss,
        ))

        return terms
