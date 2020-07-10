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
        self._optimizer = Optimizer(
            self._optimizer, self.q, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._epsilon)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        # compute target returns
        next_action = self.q.action(next_obs, self.K)
        _, next_qtv, _ = self.target_q.value(next_obs, self.N_PRIME, next_action)

        reward = reward[:, None, None]
        discount = discount[:, None, None]
        if not isinstance(steps, int):
            steps = steps[:, None, None]
        returns = n_step_target(reward, next_qtv, discount, self._gamma, steps, self._tbo)

        with tf.GradientTape() as tape:
            tau_hat, qtv, q = self.q.value(obs, self.N, action)
            error = returns - qtv   # [B, N, N']
            
            # loss
            tau_hat = tf.transpose(tf.reshape(tau_hat, [self.N, self._batch_size, 1]), [1, 0, 2]) # [B, N, 1]
            weight = tf.abs(tau_hat - tf.cast(error < 0, tf.float32))        # [B, N, N']
            huber = huber_loss(error, threshold=self.KAPPA)             # [B, N, N']
            qr_loss = tf.reduce_sum(tf.reduce_mean(weight * huber, axis=2), axis=1) # [B]
            loss = tf.reduce_mean(qr_loss)

        if self._is_per:
            error = tf.reduce_max(tf.reduce_mean(tf.abs(error), axis=2), axis=1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            loss=loss,
        ))

        return terms

