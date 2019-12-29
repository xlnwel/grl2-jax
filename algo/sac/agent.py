import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target, transformed_n_step_target
from utility.schedule import TFPiecewiseSchedule
from utility.timer import Timer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config


class Agent(BaseAgent):
    @agent_config
    def __init__(self, 
                *,
                name, 
                config, 
                models,
                dataset,
                env):
        # dataset for optimizing input pipline
        self.dataset = dataset
        
        # optimizer
        if self.optimizer.lower() == 'adam':
            Optimizer = tf.keras.optimizers.Adam
        elif self.optimizer.lower() == 'rmsprop':
            Optimizer = tf.keras.optimizers.RMSprop
        else:
            raise NotImplementedError()
        if getattr(self, 'schedule_lr', False):
            self.actor_lr = TFPiecewiseSchedule(
                [(5e5, self.actor_lr), (1.5e6, 1e-5)])
            self.q_lr = TFPiecewiseSchedule(
                [(5e5, self.q_lr), (1.5e6, 3e-5)])

        self.actor_opt = Optimizer(learning_rate=self.actor_lr,
                                    epsilon=self.epsilon)
        self.q_opt = Optimizer(learning_rate=self.q_lr,
                                epsilon=self.epsilon)
        self.ckpt_models['actor_opt'] = self.actor_opt
        self.ckpt_models['q_opt'] = self.q_opt
        if not isinstance(self.temperature, float):
            if getattr(self, 'schedule_lr', False):
                self.temp_lr = TFPiecewiseSchedule(
                    [(5e5, self.temp_lr), (1.5e6, 1e-5)])
            self.temp_opt = Optimizer(learning_rate=self.temp_lr,
                                    epsilon=self.epsilon)
            self.ckpt_models['temp_opt'] = self.temp_opt

        self.action_dim = env.action_dim
        self.is_action_discrete = env.is_action_discrete

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = [
            ([1], tf.float32, 'IS_ratio'),
            (env.state_shape, tf.float32, 'state'),
            (env.action_shape, env.action_dtype, 'action'),
            ([1], tf.float32, 'reward'),
            (env.state_shape, tf.float32, 'next_state'),
            ([1], tf.float32, 'done'),
            ([1], tf.float32, 'steps'),
        ]
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def learn_log(self, step=None):
        if step:
            self.global_steps.assign(step)
        with Timer(f'{self.model_name}: sample', 10000):
            data = self.dataset.sample()
        if self.dataset.buffer_type() != 'uniform':
            saved_indices = data['saved_indices']
            del data['saved_indices']

        terms = self.learn(**data)

        if self.dataset.buffer_type() != 'uniform':
            self.dataset.update_priorities(terms['priority'].numpy(), saved_indices.numpy())
        self.store(
            **dict((k, v.numpy()) for k, v in terms.items())
        )

    @tf.function
    def _learn(self, IS_ratio, state, action, reward, next_state, done, steps):
        if self.is_action_discrete:
            action = tf.one_hot(action, self.action_dim)
        if isinstance(self.temperature, float):
            temp = tf.convert_to_tensor(self.temperature)
        else:
            with tf.name_scope('temp_update'):
                temp, temp_grads = self._compute_temp_grads(state, IS_ratio)
                if hasattr(self, 'clip_norm'):
                    temp_grads, temp_norm = tf.clip_by_global_norm(temp_grads, self.clip_norm)
                self.temp_opt.apply_gradients(zip(temp_grads, self.temperature.trainable_variables))
        with tf.name_scope('actor_update'):
            entropy, actor_loss, actor_grads = self._compute_actor_grads(state, IS_ratio)
            if hasattr(self, 'clip_norm'):
                actor_grads, actor_norm = tf.clip_by_global_norm(actor_grads, self.clip_norm)
            self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        with tf.name_scope('q_update'):
            priority, q1, q1_loss, q_loss, q_grads = self._compute_q_grads(
                state, action, next_state, reward, 
                done, steps, IS_ratio)
            if hasattr(self, 'clip_norm'):
                q_grads, q_norm = tf.clip_by_global_norm(q_grads, self.clip_norm)
            self.q_opt.apply_gradients(
                zip(q_grads, self.q1.trainable_variables + self.q2.trainable_variables))

        self._update_target_nets()

        return dict(
            temp=temp, 
            entropy=entropy, 
            actor_loss=actor_loss, 
            q1=q1, 
            q1_loss=q1_loss, 
            q_loss=q_loss, 
            priority=priority
        )

    def _compute_temp_grads(self, state, IS_ratio):
        target_entropy = getattr(self, 'target_entropy', -self.action_dim)
        with tf.GradientTape() as tape:
            action, logpi, _ = self.actor.train_step(state)
            log_temp, temp = self.temperature.train_step(state, action)

            with tf.name_scope('temp_loss'):
                loss = -tf.reduce_mean(log_temp 
                                * tf.stop_gradient(logpi + target_entropy))
            
        with tf.name_scope('temp_grads'):
            grads = tape.gradient(loss, self.temperature.trainable_variables)

        return temp, grads

    def _compute_actor_grads(self, state, IS_ratio):
        with tf.GradientTape() as tape:
            action, logpi, entropy = self.actor.train_step(state)
            if isinstance(self.temperature, float):
                temp = self.temperature
            else:
                _, temp = self.temperature.train_step(state, action)
            q1_with_actor = self.q1.train_value(state, action)

            with tf.name_scope('actor_loss'):
                loss = tf.reduce_mean(
                    (temp * logpi - q1_with_actor))

        with tf.name_scope('actor_grads'):
            grads = tape.gradient(loss, self.actor.trainable_variables)

        return entropy, loss, grads

    def _compute_q_grads(self, state, action, next_state, reward, done, steps, IS_ratio):
        with tf.GradientTape() as tape:
            next_action, next_logpi, _ = self.actor.train_step(next_state)
            next_q1_with_actor = self.target_q1.train_value(next_state, next_action)
            next_q2_with_actor = self.target_q2.train_value(next_state, next_action)
            next_q_with_actor = tf.minimum(next_q1_with_actor, next_q2_with_actor)
            if isinstance(self.temperature, float):
                next_temp = self.temperature
            else:
                _, next_temp = self.temperature.train_step(next_state, next_action)
        
            q1 = self.q1.train_value(state, action)
            q2 = self.q2.train_value(state, action)
            with tf.name_scope('q_loss'):
                nth_value = tf.subtract(
                    next_q_with_actor, next_temp * next_logpi, name='nth_value')
                
                target_fn = transformed_n_step_target if getattr(self, 'tbo', False) else n_step_target
                target_q = target_fn(
                    reward, done, nth_value, self.gamma, steps)
                q1_error = tf.abs(target_q - q1, name='q1_error')
                q2_error = tf.abs(target_q - q2, name='q2_error')

                q1_loss = tf.reduce_mean(q1_error**2)
                q2_loss = tf.reduce_mean(q2_error**2)
                loss = q1_loss + q2_loss
        
        with tf.name_scope('q_grads'):
            grads = tape.gradient(loss, 
                self.q1.trainable_variables + self.q2.trainable_variables)

        if self.dataset.buffer_type() != 'uniform':
            priority = self._compute_priority((q1_error + q2_error) / 2.)

            return priority, q1, q1_loss, loss, grads
        else:
            return 1, q1, q1_loss, loss, grads

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        with tf.name_scope('priority'):
            priority += self.per_epsilon
            priority **= self.per_alpha
        
        return priority

    def _sync_target_nets(self):
        self.target_q1.set_weights(self.q1.get_weights())
        self.target_q2.set_weights(self.q2.get_weights())

    def _update_target_nets(self):
        tvars = self.target_q1.trainable_variables + self.target_q2.trainable_variables
        mvars = self.q1.trainable_variables + self.q2.trainable_variables
        [tvar.assign(self.polyak * tvar + (1. - self.polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]
