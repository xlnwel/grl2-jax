import tensorflow as tf

from utility.rl_utils import n_step_target, quantile_regression_loss
from utility.tf_utils import explained_variance
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

        # compute target q_target
        next_x = self.target_encoder(next_obs, training=False)
        _, next_qt_embed = self.quantile(next_x, self.N)
        next_qtvs, next_qs = self.target_q(next_x, next_qt_embed, return_value=True)
        next_qtv_v = self.target_q.v(next_qtvs)

        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        q_target = n_step_target(reward, next_qtv_v, discount, self._gamma, steps)
        q_target_ext = tf.expand_dims(q_target, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv_v, (None, self.N)],
            [q_target_ext, (None, 1, self.N)],
        ])

        with tf.GradientTape() as tape:
            x = self.encoder(obs, training=True)
            tau_hat, qt_embed = self.quantile(x, self.N)
            qtvs = self.q(x, qt_embed)
            action_ext = tf.expand_dims(action, axis=1)
            qtv_ext = tf.reduce_sum(qtvs * action_ext, axis=-1, keepdims=True)  # [B, N, 1]
            error, qr_loss = quantile_regression_loss(
                qtv_ext, q_target_ext, tau_hat, kappa=self.KAPPA, return_error=True)
            loss = tf.reduce_mean(IS_ratio * qr_loss)

        terms['norm'] = self._optimizer(tape, loss)

        qs = self.q.value(qtvs)
        v = self.q.v(qs)
        logits = self.q.logits(qs, v)
        probs, logps = self.q.prob_logp(logits)
        probs_avg = tf.reduce_mean(probs, 0)
        self.q.update_prior(probs_avg, self._prior_lr)
        if self._is_per:
            # TODO: compute priority using loss
            error = tf.abs(error)
            error = tf.reduce_max(tf.reduce_mean(error, axis=-1), axis=-1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        entropy = -tf.reduce_sum(probs * logps, axis=-1)
        q = tf.reduce_mean(qtv_ext, axis=(1, 2))
        q_target = tf.reduce_mean(q_target, axis=1)
        terms.update(dict(
            entropy=entropy,
            entropy_max=tf.reduce_max(entropy),
            entropy_min=tf.reduce_min(entropy),
            v=tf.reduce_mean(v),
            q=tf.reduce_mean(q),
            qr_loss=qr_loss,
            loss=loss,
            explained_variance_q=explained_variance(q_target, q),
        ))

        return terms

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_encoder.variables + self.target_q.variables + self.target_quantile.variables
        mvars = self.encoder.variables + self.q.variables + self.quantile.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]
