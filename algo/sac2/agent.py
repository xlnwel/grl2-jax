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
from algo.dqn.base import get_data_format


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._is_per = self._replay_type.endswith('per')
        self.dataset = dataset

        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule([(2e5, self._actor_lr), (1e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule([(2e5, self._q_lr), (1e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._value_opt = Optimizer(self._optimizer, self.value, self._q_lr)
        self._q_opt = Optimizer(self._optimizer, [self.q, self.q2], self._q_lr)

        if self.temperature.is_trainable():
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
        if getattr(self, 'n_steps', 1) > 1:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def __call__(self, obs, evaluation=False, **kwargs):
        return self.model.action(
            obs, 
            evaluation=evaluation, 
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
        q_value = lambda q, obs, act: q(tf.concat([obs, act], -1)).mode()
        with tf.GradientTape() as actor_tape:
            act_dist, terms = self.actor.step(obs)
            new_action = act_dist.sample(one_hot=True)
            new_logpi = act_dist.log_prob(new_action)
            if isinstance(self.temperature, (float, tf.Variable)):
                temp = self.temperature
            else:
                _, temp = self.temperature(obs, new_action)
            q_with_actor = q_value(self.q, obs, new_action)
            q2_with_actor = q_value(self.q2, obs, new_action)
            q_with_actor = tf.minimum(q_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(IS_ratio * 
                (temp * new_logpi - q_with_actor))
        
        if self.temperature.is_trainable():
            with tf.GradientTape() as temp_tape:
                log_temp, temp = self.temperature(obs, new_action)
                temp_loss = -tf.reduce_mean(IS_ratio * log_temp 
                    * tf.stop_gradient(new_logpi + self._target_entropy))
                terms['temp_loss'] = temp_loss
                terms['temp'] = temp

        with tf.GradientTape() as value_tape:
            value = self.value(obs).mode()
            target_value = q_with_actor - temp * new_logpi
            value_loss = .5 * tf.reduce_mean(IS_ratio * (target_value - value)**2)

        with tf.GradientTape() as q_tape:
            q = q_value(self.q, obs, action)
            q2 = q_value(self.q2, obs, action)
            q = tf.minimum(q, q2)
            next_value = self.target_value(next_obs).mode()
            
            target_q = n_step_target(reward, next_value, discount, self._gamma, steps)
            target_q = tf.stop_gradient(target_q)
            q_error = target_q - q
            q2_error = target_q - q2
            q_loss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
            q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
            q_loss = q_loss + q2_loss

        if self._is_per:
            priority = self._compute_priority((tf.abs(q_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority

        terms['actor_norm'] = self._actor_opt(actor_tape, actor_loss)
        terms['value_norm'] = self._value_opt(value_tape, value_loss)
        terms['q_norm'] = self._q_opt(q_tape, q_loss)
        if not isinstance(self.temperature, (float, tf.Variable)):
            terms['temp_norm'] = self._temp_opt(temp_tape, temp_loss)
            
        terms.update(dict(
            actor_loss=actor_loss,
            q=q, 
            q2=q2,
            logpi=new_logpi,
            action_entropy=act_dist.entropy(),
            target_q=target_q,
            q_loss=q_loss, 
            q2_loss=q2_loss,
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
        [tvar.assign(mvar) for tvar, mvar in zip(
            self.target_value.variables, self.value.variables)]

    @tf.function
    def _update_target_nets(self):
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(
                self.target_value.trainable_variables, self.value.trainable_variables)]
