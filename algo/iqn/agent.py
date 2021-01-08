import tensorflow as tf

from utility.rl_loss import n_step_target, quantile_regression_loss
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from core.decorator import override
from algo.dqn.base import DQNBase, get_data_format


class Agent(DQNBase):
    @override(DQNBase)
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule([(5e5, self._lr), (2e6, 5e-5)])
        models = [self.encoder, self.quantile, self.q]
        self._optimizer = Optimizer(self._optimizer, models, 
            self._lr, epsilon=self._epsilon)

    @override(DQNBase)
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        # compute the target
        next_x = self.encoder(next_obs)
        _, next_qt_embed = self.quantile(next_x, self.N)
        next_action = self.q.action(next_x, next_qt_embed)
        next_x = self.target_encoder(next_obs)
        _, next_qt_embed = self.target_quantile(next_x, self.N_PRIME)
        next_qtv= self.target_q(next_x, next_qt_embed, next_action)
        reward = reward[:, None]
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
            x = tf.expand_dims(x, 1)
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
