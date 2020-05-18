import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from utility.display import pwc
from utility.rl_utils import n_step_target, transformed_n_step_target
from utility.utils import Every
from utility.losses import huber_loss
from utility.schedule import TFPiecewiseSchedule, PiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from core.optimizer import Optimizer


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._dtype = global_policy().compute_dtype
        if dataset is None:
            self._is_per = False
            is_nsteps = False
        else:
            self._is_per = dataset.buffer_type().endswith('proportional')
            is_nsteps = 'steps' in dataset.data_format
            
        self.dataset = dataset

        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)
        if self._schedule_eps:
            self._act_eps = PiecewiseSchedule(((5e4, 1), (4e5, .02)))

        self._to_summary = Every(100000)
        self._to_sync = Every(self._target_update_freq)
        # optimizer
        self._optimizer = Optimizer(self._optimizer, self.q, self._lr, clip_norm=self._clip_norm)
        self._ckpt_models['optimizer'] = self._optimizer

        self._action_dim = env.action_dim

        # Explicitly instantiate tf.function to initialize variables
        obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else self._dtype
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((env.action_dim,), self._dtype, 'action'),
            reward=((), self._dtype, 'reward'),
            nth_obs=(env.obs_shape, env.obs_dtype, 'nth_obs'),
            discount=((), self._dtype, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), self._dtype, 'IS_ratio')
        if is_nsteps:
            TensorSpecs['steps'] = ((), self._dtype, 'steps')
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def reset_noisy(self):
        self.q.reset_noisy()

    def __call__(self, obs, deterministic=False):
        if self._schedule_eps:
            eps = self._act_eps.value(self.global_steps.numpy())
            self.store(act_eps=eps)
        else:
            eps = self._act_eps
        return self.q(obs, deterministic, eps)

    def learn_log(self, step):
        self.global_steps.assign(step)
        with TBTimer('sample', 2500):
            data = self.dataset.sample()

        if self._is_per:
            idxes = data['idxes'].numpy()
            del data['idxes']
        with TBTimer('learn', 2500):
            terms = self.learn(**data)
        if self._to_sync(step):
            self._sync_target_nets()

        if self._schedule_lr:
            step = tf.convert_to_tensor(step, tf.float32)
            terms['lr'] = self._lr(step)
        terms = {k: v.numpy() for k, v in terms.items()}

        if self._is_per:
            self.dataset.update_priorities(terms['priority'], idxes)
        self.store(**terms)

    @tf.function
    def _learn(self, obs, action, reward, nth_obs, discount, steps=1, IS_ratio=1):
        target_fn = (transformed_n_step_target if self._tbo 
                    else n_step_target)
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        with tf.GradientTape() as tape:
            q = self.q.value(obs, action)
            nth_action = self.q.action(nth_obs, noisy=False)
            nth_action = tf.one_hot(nth_action, self._action_dim, dtype=self._dtype)
            nth_q = self.target_q.value(nth_obs, nth_action, noisy=False)
            target_q = target_fn(reward, nth_q, discount, self._gamma, steps)
            target_q = tf.stop_gradient(target_q)
            error = target_q - q
            loss = tf.reduce_mean(IS_ratio * loss_fn(error))

        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            loss=loss,
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    @tf.function
    def _sync_target_nets(self):
        [tv.assign(mv) for mv, tv in zip(
            self.q.trainable_variables, self.target_q.trainable_variables)]
