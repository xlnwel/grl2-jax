import tensorflow as tf

from utility.rl_utils import n_step_target, quantile_regression_loss
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from algo.dqn.base import DQNBase, get_data_format


class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule([(5e5, self._lr), (2e6, 5e-5)])
        models = [self.encoder, self.quantile, self.q]
        self._optimizer = Optimizer(self._optimizer, models, 
            self._lr, epsilon=self._epsilon)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        # compute target returns
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
        returns = n_step_target(reward, next_qtv, discount, self._gamma, steps)
        returns = tf.expand_dims(returns, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv, (None, self.N_PRIME)],
            [returns, (None, 1, self.N_PRIME)],
        ])

        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            tau_hat, qt_embed = self.quantile(x, self.N)
            x = tf.expand_dims(x, 1)
            qtv = self.q(x, qt_embed, action)
            qtv = tf.expand_dims(qtv, axis=-1)  # [B, N, 1]
            qr_loss = quantile_regression_loss(qtv, returns, tau_hat, kappa=self.KAPPA)
            loss = tf.reduce_mean(IS_ratio * qr_loss)

        terms['norm'] = self._optimizer(tape, loss)

        if self._is_per:
            priority = self._compute_priority(qr_loss)
            terms['priority'] = priority
        
        terms.update(dict(
            returns=returns,
            q=tf.reduce_mean(qtv),
            qr_loss=qr_loss,
            loss=loss,
        ))

        return terms

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_encoder.variables + self.target_q.variables + self.target_quantile.variables
        mvars = self.encoder.variables + self.q.variables + self.quantile.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]
