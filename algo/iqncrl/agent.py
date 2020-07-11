import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target, huber_loss
from utility.utils import Every
from utility.schedule import TFPiecewiseSchedule, PiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.dqn.agent import get_data_format, DQNBase


class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)
        self._optimizer = Optimizer(
            self._optimizer, [self.q, self.crl], self._lr, 
            clip_norm=self._clip_norm, epsilon=self._epsilon)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        # compute target returns
        next_action = self.q.action(next_obs, self.K)
        _, _, next_qtv, _ = self.target_q.value(next_obs, self.N_PRIME, next_action)
        reward = reward[:, None, None]
        discount = discount[:, None, None]
        if not isinstance(steps, int):
            steps = steps[:, None, None]
        returns = n_step_target(reward, next_qtv, discount, self._gamma, steps, self._tbo)
        returns = tf.transpose(returns, (0, 2, 1))      # [B, 1, N']

        with tf.GradientTape() as tape:
            z, tau_hat, qtv, q = self.q.value(obs, self.N, action)
            error = returns - qtv   # [B, N, N']
            tf.debugging.assert_shapes([
                [returns, (None, 1, self.N_PRIME)],
                [qtv, (None, self.N, 1)],
                [error, (None, self.N, self.N_PRIME)],
            ])
            # loss
            tau_hat = tf.transpose(tf.reshape(tau_hat, [self.N, self._batch_size, 1]), [1, 0, 2]) # [B, N, 1]
            weight = tf.abs(tau_hat - tf.cast(error < 0, tf.float32))        # [B, N, N']
            huber = huber_loss(error, threshold=self.KAPPA)             # [B, N, N']
            qr_loss = tf.reduce_sum(tf.reduce_mean(weight * huber, axis=2), axis=1) # [B]
            qr_loss = tf.reduce_mean(qr_loss)

            z_pos = self.target_q.cnn(obs)
            z_anchor = self.crl(z)
            z_pos = self.crl(z_pos)
            logits = self.crl.logits(z_anchor, z_pos)
            tf.debugging.assert_shapes([[logits, (self._batch_size, self._batch_size)]])
            labels = tf.range(self._batch_size)
            infonce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            infonce = tf.reduce_mean(infonce)
            loss = qr_loss + self._crl_coef * infonce

        if self._is_per:
            error = tf.reduce_max(tf.reduce_mean(tf.abs(error), axis=2), axis=1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            qr_loss=qr_loss,
            infonce=infonce,
            loss=loss,
        ))

        return terms
