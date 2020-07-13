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
            self._iqn_lr = TFPiecewiseSchedule(
                [(5e5, self._iqn_lr), (2e6, 5e-5)], outside_value=5e-5)
        self._iqn_opt = Optimizer(
            self._iqn_opt, self.q, self._iqn_lr, 
            clip_norm=self._clip_norm, epsilon=self._epsilon)
        self._fpn_opt = Optimizer(self._fpn_opt, self.fpn, self._fpn_lr)

    @tf.function
    def summary(self, data, terms):
        tf.summary.histogram('tau', terms['tau'], step=self._env_step)
        tf.summary.histogram('fpn_entropy', terms['fpn_entropy'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        # compute target returns
        next_x = self.q.cnn(next_obs)
        
        next_tau, next_tau_hat, _ = self.fpn(next_x)
        next_action = self.q.action(next_x, next_tau_hat, tau_range=next_tau)
        
        next_x = self.target_q.cnn(next_obs)
        next_tau, next_tau_hat, _ = self.fpn(next_x)
        next_qtv = self.target_q.value(
            next_x, next_tau_hat, action=next_action)
        
        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        returns = n_step_target(reward, next_qtv, discount, self._gamma, steps, self._tbo)
        tf.debugging.assert_shapes([
            [next_qtv, (None, self.N)],
            [returns, (None, self.N)],
        ])
        returns = tf.expand_dims(returns, axis=1)      # [B, 1, N]

        with tf.GradientTape(persistent=True) as tape:
            x = self.q.cnn(obs)
            x_no_grad = tf.stop_gradient(x) # forbid gradients to cnn when computing fpn loss
            
            tau, tau_hat, fpn_entropy = self.fpn(x_no_grad)
            terms['tau'] = tau
            tau_hat = tf.stop_gradient(tau_hat) # forbid gradients to fpn when computing qr loss
            qtv, q = self.q.value(
                x, tau_hat, tau_range=tau, action=action)
            qtv_exp = tf.expand_dims(qtv, axis=-1)
            tau_hat = tf.expand_dims(tau_hat, -1) # [B, N, 1]
            qr_loss = quantile_regression_loss(qtv_exp, returns, tau_hat, kappa=self.KAPPA)
            qr_loss = tf.reduce_mean(qr_loss)

            # compute gradients for fpn
            tau_qtv = self.q.value(x_no_grad, tau[..., 1:-1], action=action)     # [B, N-1, A]
            tf.debugging.assert_shapes([
                [qtv, (None, self.N)],
                [tau_qtv, (None, self.N-1)],
            ])

            # we use 𝜃 to represent F^{-1} for brevity
            diff1 = tau_qtv - qtv[..., :-1]  # 𝜃(𝜏[i]) - 𝜃(\hat 𝜏[i-1])
            sign1 = tau_qtv > qtv[..., :-1]
            abs_diff1 = tf.where(sign1, diff1, -diff1)
            diff2 = tau_qtv - qtv[..., 1:]  # 𝜃(𝜏[i]) - 𝜃(\hat 𝜏[i])
            sign2 = tau_qtv > qtv[..., 1:]
            abs_diff2 = tf.where(sign2, diff2, -diff2)
            fpn_out_grads = abs_diff1 + abs_diff2
            fpn_out_grads = tf.stop_gradient(fpn_out_grads)
            fpn_raw_loss = tf.reduce_mean(fpn_out_grads * tau[..., 1:-1])
            fpn_entropy = tf.reduce_mean(fpn_entropy)
            fpn_entropy_loss = - self._ent_coef * fpn_entropy
            fpn_loss = fpn_raw_loss + fpn_entropy_loss

        if self._is_per:
            error = tf.reduce_max(tf.reduce_mean(tf.abs(error), axis=2), axis=1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        terms['iqn_norm'] = self._iqn_opt(tape, qr_loss)
        terms['fpn_norm'] = self._fpn_opt(tape, fpn_loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            qr_loss=qr_loss,
            fpn_entropy=fpn_entropy,
            fpn_out_grads=fpn_out_grads,
            fpn_raw_loss=fpn_raw_loss,
            fpn_entropy_loss=fpn_entropy_loss,
            fpn_loss=fpn_loss,
        ))

        return terms

