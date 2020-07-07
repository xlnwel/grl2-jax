import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.rl_utils import n_step_target
from utility.schedule import TFPiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.dqn.agent import get_data_format


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._is_per = self._replay_type.endswith('per')
        self.dataset = dataset

        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule(
                [(2e5, self._actor_lr), (1e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule(
                [(2e5, self._q_lr), (1e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._q_opt = Optimizer(self._optimizer, [self.q1, self.q2], self._q_lr)

        if isinstance(self.temperature, float):
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete
        if not hasattr(self, '_target_entropy'):
            self._target_entropy = .98 * np.log(self._action_dim) \
                if self._is_action_discrete else -self._action_dim

        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((env.action_dim,), tf.float32, 'action'),
            reward=((), tf.float32, 'reward'),
            next_obs=(env.obs_shape, env.obs_dtype, 'next_obs'),
            discount=((), tf.float32, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if 'steps'  in self.dataset.data_format:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def __call__(self, obs, deterministic=False, **kwargs):
        return self.model.action(
            obs, 
            deterministic=deterministic, 
            epsilon=self._act_eps).numpy()

    @step_track
    def learn_log(self, step):
        data = self.dataset.sample()
        if self._is_per:
            idxes = data.pop('idxes').numpy()

        terms = self.learn(**data)
        self._update_target_nets()

        if self._schedule_lr:
            step = tf.convert_to_tensor(step, tf.float32)
            terms['actor_lr'] = self._actor_lr(step)
            terms['q_lr'] = self._q_lr(step)
        terms = {k: v.numpy() for k, v in terms.items()}

        if self._is_per:
            self.dataset.update_priorities(terms['priority'], idxes)
        self.store(**terms)
        return 1

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        next_action, next_logpi, _ = self.actor.train_step(next_obs)
        next_q1_with_actor = self.target_q1(next_obs, next_action)
        next_q2_with_actor = self.target_q2(next_obs, next_action)
        next_q_with_actor = tf.minimum(next_q1_with_actor, next_q2_with_actor)
        if isinstance(self.temperature, (tf.Variable)):
            temp = self.temperature
        else:
            _, temp = self.temperature()
        next_value = next_q_with_actor - temp * next_logpi
        target_q = n_step_target(reward, next_value, discount, self._gamma, steps)

        terms = {}
        with tf.GradientTape() as tape:
            q1 = self.q1(obs, action)
            q2 = self.q2(obs, action)
            q1_error = target_q - q1
            q2_error = target_q - q2
            q1_loss = .5 * tf.reduce_mean(IS_ratio * q1_error**2)
            q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
            q_loss = q1_loss + q2_loss
        terms['q_norm'] = self._q_opt(tape, q_loss)

        with tf.GradientTape(persistent=True) as actor_tape:
            action, logpi, actor_terms = self.actor.train_step(obs)
            terms.update(actor_terms)
            log_temp, temp = self.temperature(obs, action)
            temp_loss = -tf.reduce_mean(IS_ratio * log_temp 
                * tf.stop_gradient(logpi + self._target_entropy))
            q1_with_actor = self.q1(obs, action)
            q2_with_actor = self.q2(obs, action)
            q_with_actor = tf.minimum(q1_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(IS_ratio * 
                (tf.stop_gradient(temp) * logpi - q_with_actor))

        self._actor_opt(actor_tape, actor_loss)
        self._temp_opt(actor_tape, temp_loss)

        if self._is_per:
            priority = self._compute_priority((tf.abs(q1_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority
            
        terms.update(dict(
            temp=temp,
            temp_loss=temp_loss,
            actor_loss=actor_loss,
            q1=q1, 
            q2=q2,
            logpi=logpi,
            target_q=target_q,
            q1_loss=q1_loss, 
            q2_loss=q2_loss,
            q_loss=q_loss, 
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        tf.debugging.assert_greater(priority, 0.)
        return priority

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_q1.variables + self.target_q2.variables
        mvars = self.q1.variables + self.q2.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]

    @tf.function
    def _update_target_nets(self):
        tvars = self.target_q1.trainable_variables + self.target_q2.trainable_variables
        mvars = self.q1.trainable_variables + self.q2.trainable_variables
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]
