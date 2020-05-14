import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.rl_utils import n_step_target
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
            self._actor_lr = TFPiecewiseSchedule(
                [(2e5, self._actor_lr), (1e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule(
                [(2e5, self._q_lr), (1e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._value_opt = Optimizer(self._optimizer, self.value, self._q_lr)
        self._q_opt = Optimizer(self._optimizer, [self.q1, self.q2], self._q_lr)
        self._ckpt_models['actor_opt'] = self._actor_opt
        self._ckpt_models['value_opt'] = self._value_opt
        self._ckpt_models['q_opt'] = self._q_opt

        if isinstance(self.temperature, float):
            # convert to variable, useful for scheduling
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            if getattr(self, '_schedule_lr', False):
                self._temp_lr = TFPiecewiseSchedule(
                    [(5e5, self._temp_lr), (1e6, 1e-5)])
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)
            self._ckpt_models['temp_opt'] = self._temp_opt

        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        TensorSpecs = dict(
            obs=(env.obs_shape, self._dtype, 'obs'),
            action=((env.action_dim,), self._dtype, 'action'),
            reward=((), self._dtype, 'reward'),
            nth_obs=(env.obs_shape, self._dtype, 'nth_obs'),
            discount=((), self._dtype, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), self._dtype, 'IS_ratio')
        if 'steps'  in self.dataset.data_format:
            TensorSpecs['steps'] = ((), self._dtype, 'steps')
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def __call__(self, obs, deterministic=False, epsilon=0):
        if len(obs.shape) % 2 == 1:
            obs = np.expand_dims(obs, 0)
        return np.squeeze(self.action(obs).numpy())
    
    @tf.function
    def action(self, obs, deterministic=False, epsilon=0):
        if deterministic:
            action = self.actor(obs)[0].mode()
        else:
            act_dist = self.actor(obs)[0]
            action = act_dist.sample()
            if epsilon:
                action = tf.clip_by_value(
                    tfd.Normal(action, self._act_eps).sample(), -1, 1)
        
        return action

    def learn_log(self, step):
        with TBTimer('sample', 1000):
            data = self.dataset.sample()
        if self._is_per:
            idxes = data['idxes'].numpy()
            del data['idxes']

        with TBTimer('learn', 1000):
            terms = self.learn(**data)
        self._update_target_nets()

        if self._schedule_lr:
            step = tf.convert_to_tensor(step, tf.float32)
            terms['actor_lr'] = self._actor_lr(step)
            terms['q_lr'] = self._q_lr(step)
            if not isinstance(self.temperature, (float, tf.Variable)):
                terms['temp_lr'] = self._temp_lr(step)
        terms = {k: v.numpy() for k, v in terms.items()}

        if self._is_per:
            self.dataset.update_priorities(terms['priority'], idxes)
        self.store(**terms)

    @tf.function
    def _learn(self, obs, action, reward, nth_obs, discount, steps=1, IS_ratio=1):
        target_entropy = getattr(self, 'target_entropy', -self._action_dim)
        q_value = lambda q, obs, act: q(tf.concat([obs, act], -1)).mode()
        with tf.GradientTape() as actor_tape:
            act_dist, terms = self.actor(obs)
            new_action = act_dist.sample()
            new_logpi = act_dist.log_prob(new_action)
            if isinstance(self.temperature, (float, tf.Variable)):
                temp = self.temperature
            else:
                _, temp = self.temperature(obs, new_action)
            q1_with_actor = q_value(self.q1, obs, new_action)
            q2_with_actor = q_value(self.q2, obs, new_action)
            q_with_actor = tf.minimum(q1_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(IS_ratio * 
                (temp * new_logpi - q_with_actor))
        
        with tf.GradientTape() as temp_tape:
            if isinstance(self.temperature, (float, tf.Variable)):
                temp = self.temperature
            else:
                log_temp, temp = self.temperature(obs, new_action)
                temp_loss = -tf.reduce_mean(IS_ratio * log_temp 
                    * tf.stop_gradient(new_logpi + target_entropy))
                terms['temp'] = temp
                terms['temp_loss'] = temp_loss

        with tf.GradientTape() as value_tape:
            value = self.value(obs).mode()
            target_value = q_with_actor - temp * new_logpi
            value_loss = .5 * tf.reduce_mean(IS_ratio * (target_value - value)**2)

        with tf.GradientTape() as q_tape:
            q1 = q_value(self.q1, obs, action)
            q2 = q_value(self.q2, obs, action)
            q = tf.minimum(q1, q2)
            nth_value = self.target_value(nth_obs).mode()
            
            target_q = n_step_target(reward, nth_value, discount, self._gamma, steps)
            target_q = tf.stop_gradient(target_q)
            q1_error = target_q - q1
            q2_error = target_q - q2
            q1_loss = .5 * tf.reduce_mean(IS_ratio * q1_error**2)
            q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
            q_loss = q1_loss + q2_loss

        if self._is_per:
            priority = self._compute_priority((tf.abs(q1_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority

        terms['actor_norm'] = self._actor_opt(actor_tape, actor_loss)
        terms['value_norm'] = self._value_opt(value_tape, value_loss)
        terms['q_norm'] = self._q_opt(q_tape, q_loss)
        if not isinstance(self.temperature, (float, tf.Variable)):
            terms['temp_norm'] = self._temp_opt(temp_tape, temp_loss)
            
        terms.update(dict(
            actor_loss=actor_loss,
            q1=q1, 
            q2=q2,
            logpi=new_logpi,
            action_entropy=act_dist.entropy(),
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
        [tvar.assign(mvar) for tvar, mvar in zip(
            self.target_value.variables, self.value.variables)]

    @tf.function
    def _update_target_nets(self):
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(
                self.target_value.trainable_variables, self.value.trainable_variables)]
