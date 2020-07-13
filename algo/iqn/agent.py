import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target, huber_loss, quantile_regression_loss
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
            self._optimizer, self.q, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._epsilon)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        # compute target returns
        next_action = self.q.action(next_obs, self.K)
        _, next_qtv, _ = self.target_q.value(next_obs, self.N_PRIME, next_action)
        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        returns = n_step_target(reward, next_qtv, discount, self._gamma, steps, self._tbo)
        returns = tf.expand_dims(returns, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv, (None, self.N_PRIME)],
            [returns, (None, 1, self.N_PRIME)],
        ])

        with tf.GradientTape() as tape:
            tau_hat, qtv, q = self.q.value(obs, self.N, action)
            qtv = tf.expand_dims(qtv, axis=-1)  # [B, N, 1]
            qr_loss = quantile_regression_loss(qtv, returns, tau_hat, kappa=self.KAPPA)
            loss = tf.reduce_mean(IS_ratio * qr_loss)

        if self._is_per:
            priority = self._compute_priority(qr_loss)
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            loss=loss,
        ))

        return terms
