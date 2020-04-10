import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from utility.display import pwc
from utility.rl_utils import n_step_target, transformed_n_step_target
from utility.losses import huber_loss
from utility.schedule import TFPiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from core.optimizer import Optimizer


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._dtype = global_policy().compute_dtype
        self._is_per = not dataset.buffer_type().endswith('uniform')

        self.dataset = dataset

        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)

        # optimizer
        self._optimizer = Optimizer(self._optimizer, self.q, self._lr)
        self._ckpt_models['optimizer'] = self._optimizer

        self._action_dim = env.action_dim

        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=(env.obs_shape, self._dtype, 'obs'),
            action=((env.action_dim,), self._dtype, 'action'),
            reward=((), self._dtype, 'reward'),
            next_obs=(env.obs_shape, self._dtype, 'next_obs'),
            done=((), self._dtype, 'done'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), self._dtype, 'IS_ratio')
        if 'steps'  in self.dataset.data_format:
            TensorSpecs['steps'] = ((), self._dtype, 'steps')
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def __call__(self, obs, deterministic=False):
        return self.q(obs, deterministic, getattr(self, '_act_eps', 0))

    def learn_log(self, step):
        data = self.dataset.sample()
        if self._is_per:
            saved_idxes = data['saved_idxes'].numpy()
            del data['saved_idxes']
        terms = self.learn(**data)
        if step % self._target_update_freq == 0:
            self._sync_target_nets()

        if self._schedule_lr:
            step = tf.convert_to_tensor(step, tf.float32)
            terms['lr'] = self._lr(step)
        terms = {k: v.numpy() for k, v in terms.items()}

        if self._is_per:
            self.dataset.update_priorities(terms['priority'], saved_idxes)
        self.store(**terms)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, done, steps=1, IS_ratio=1):
        target_fn = (transformed_n_step_target if self._tbo 
                    else n_step_target)
        loss_fn = huber_loss if self._loss_type == 'huber' else tf.square
        terms = {}
        with tf.GradientTape() as tape:
            q = self.q.value(obs, action)
            nth_action = tf.one_hot(self.q.action(next_obs, noisy=False), self._action_dim)
            nth_q = self.target_q.value(next_obs, nth_action)
            target_q = target_fn(reward, done, nth_q, self._gamma, steps)
            error = target_q - q
            loss = .5 * tf.reduce_mean(IS_ratio * loss_fn(error))
        
        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)

        terms.update(dict(
            q=q,
            loss=loss
        ))
        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        tf.debugging.assert_greater(priority, 0.)
        return priority

    def _sync_target_nets(self):
        self.target_q.set_weights(self.q.get_weights())
