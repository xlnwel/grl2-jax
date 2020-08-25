import tensorflow as tf

from utility.rl_utils import n_step_target, huber_loss
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from algo.dqn.base import DQNBase, get_data_format

class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule( [(5e5, self._lr), (2e6, 5e-5)])
        self._optimizer = Optimizer(self._optimizer, self.q, self._lr, clip_norm=self._clip_norm)

    def reset_noisy(self):
        self.q.reset_noisy()

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        # compute target returns
        next_action = self.q.action(next_obs, noisy=False)
        next_q = self.target_q.value(next_obs, next_action, noisy=False)
        returns = n_step_target(reward, next_q, discount, self._gamma, steps, self._tbo)

        with tf.GradientTape() as tape:
            q = self.q.value(obs, action)
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

    @tf.function
    def _sync_target_nets(self):
        [tv.assign(mv) for mv, tv in zip(
            self.q.trainable_variables, self.target_q.trainable_variables)]
