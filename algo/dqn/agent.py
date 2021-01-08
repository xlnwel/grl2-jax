import tensorflow as tf

from utility.rl_utils import n_step_target, huber_loss
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from algo.dqn.base import DQNBase, get_data_format

class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule( [(5e5, self._lr), (2e6, 5e-5)])
        models = [self.encoder, self.q]
        self._optimizer = Optimizer(
            self._optimizer, models, self._lr, clip_norm=self._clip_norm)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        # compute target returns
        next_x = self.encoder(next_obs)
        next_action = self.q.action(next_x, noisy=False)
        next_x = self.target_encoder(next_obs)
        next_q = self.target_q(next_x, next_action, noisy=False)
        returns = n_step_target(reward, next_q, discount, self._gamma, steps, self._tbo)

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
