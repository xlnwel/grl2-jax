import numpy as np
import tensorflow as tf

from utility.display import pwc
from core.tf_config import build
from utility.tf_utils import n_step_target
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
                state_shape,
                state_dtype, 
                action_dim,
                action_dtype):
        # dataset for optimizing input pipline
        self.dataset = dataset
        
        # optimizer
        if self.optimizer.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam
        elif self.optimizer.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop
        else:
            raise NotImplementedError()
        self.actor_opt = optimizer(learning_rate=self.actor_lr,
                                    epsilon=self.epsilon)
        self.softq_opt = optimizer(learning_rate=self.softq_lr,
                                epsilon=self.epsilon)
        self.temp_opt = optimizer(learning_rate=self.temp_lr,
                                epsilon=self.epsilon)
        self.ckpt_models['actor_opt'] = self.actor_opt
        self.ckpt_models['softq_opt'] = self.softq_opt
        self.ckpt_models['temp_opt'] = self.temp_opt

        # for temperature loss
        self.action_dim = action_dim
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = [
            ([1], tf.float32, 'IS_ratio'),
            (state_shape, state_dtype, 'state'),
            ([action_dim], action_dtype, 'action'),
            ([1], tf.float32, 'reward'),
            (state_shape, state_dtype, 'next_state'),
            ([1], tf.float32, 'done'),
            ([1], tf.float32, 'steps'),
        ]
        self.train_step = build(self._train_step, TensorSpecs)

    def train_log(self):
        data = self.dataset.get_data()
        saved_indices = data['saved_indices']
        del data['saved_indices']
        # tf.summary.trace_on()
        temp, actor_loss, q1, q1_loss, softq_loss, priority = self.train_step(**data)
        # tf.summary.trace_export('train')
        self.dataset.update_priorities(priority.numpy(), saved_indices.numpy())
        self.store(
            temp=temp.numpy(), 
            actor_loss=actor_loss.numpy(),
            q1=q1.numpy(),
            q1_loss=q1_loss.numpy(),
            softq_loss=softq_loss.numpy(),
            priority=priority.numpy(),
        )

    @tf.function
    def _train_step(self, IS_ratio, state, action, reward, next_state, done, steps):
        with tf.name_scope('temp_train'):
            temp, temp_grads = self._compute_temp_grads(state, IS_ratio)
            if hasattr(self, 'clip_norm'):
                temp_grads, temp_norm = tf.clip_by_global_norm(temp_grads, self.clip_norm)
            self.temp_opt.apply_gradients(zip(temp_grads, self.temperature.trainable_variables))
        with tf.name_scope('actor_train'):
            actor_loss, actor_grads = self._compute_actor_grads(state, IS_ratio)
            if hasattr(self, 'clip_norm'):
                actor_grads, actor_norm = tf.clip_by_global_norm(actor_grads, self.clip_norm)
            self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        with tf.name_scope('softq_train'):
            priority, q1, q1_loss, softq_loss, softq_grads = self._compute_softq_grads(
                state, action, next_state, reward, 
                done, steps, IS_ratio)
            if hasattr(self, 'clip_norm'):
                softq_grads, softq_norm = tf.clip_by_global_norm(softq_grads, self.clip_norm)
            self.softq_opt.apply_gradients(
                zip(softq_grads, self.softq1.trainable_variables + self.softq2.trainable_variables))

        self._update_target_nets()

        return temp, actor_loss, q1, q1_loss, softq_loss, priority

    def _compute_temp_grads(self, state, IS_ratio):
        target_entropy = -self.action_dim
        with tf.GradientTape() as tape:
            action, logpi = self.actor.train_step(state)
            log_temp, temp = self.temperature.train_step(state, action)

            with tf.name_scope('temp_loss'):
                loss = -tf.reduce_mean(IS_ratio * log_temp 
                                * tf.stop_gradient(logpi + target_entropy))
            
        with tf.name_scope('temp_grads'):
            grads = tape.gradient(loss, self.temperature.trainable_variables)

        return temp, grads

    def _compute_actor_grads(self, state, IS_ratio):
        with tf.GradientTape() as tape:
            action, logpi = self.actor.train_step(state)
            _, temp = self.temperature.train_step(state, action)
            q1_with_actor = self.softq1.train_step(state, action)

            with tf.name_scope('actor_loss'):
                loss = tf.reduce_mean(
                    IS_ratio * (temp * logpi - q1_with_actor))

        with tf.name_scope('actor_grads'):
            grads = tape.gradient(loss, self.actor.trainable_variables)

        return loss, grads

    def _compute_softq_grads(self, state, action, next_state, reward, done, steps, IS_ratio):
        with tf.GradientTape() as tape:
            next_action, next_logpi = self.actor.train_step(next_state)
            next_q1_with_actor = self.target_softq1.train_step(next_state, next_action)
            next_q2_with_actor = self.target_softq2.train_step(next_state, next_action)
            next_q_with_actor = tf.minimum(next_q1_with_actor, next_q2_with_actor)
            _, next_temp = self.temperature.train_step(next_state, next_action)
        
            q1 = self.softq1.train_step(state, action)
            q2 = self.softq2.train_step(state, action)
            with tf.name_scope('softq_loss'):
                nth_value = tf.subtract(next_q_with_actor, next_temp * next_logpi, name='nth_value')
                target_q = n_step_target(reward, done, 
                                        nth_value, self.gamma, steps)
                q1_error = tf.abs(target_q - q1, name='q1_error')
                q2_error = tf.abs(target_q - q2, name='q2_error')

                q1_loss = tf.reduce_mean(IS_ratio * q1_error**2)
                q2_loss = tf.reduce_mean(IS_ratio * q2_error**2)
                loss = q1_loss + q2_loss

        priority = self._compute_priority((q1_error + q2_error) / 2.)
        with tf.name_scope('softq_grads'):
            grads = tape.gradient(loss, 
                self.softq1.trainable_variables + self.softq2.trainable_variables)

        return priority, q1, q1_loss, loss, grads

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        with tf.name_scope('priority'):
            priority += self.per_epsilon
            priority **= self.per_alpha
        
        return priority

    def _initialize_target_nets(self):
        self.target_softq1.set_weights(self.soft_q1.get_weights())
        self.target_softq2.set_weights(self.soft_q2.get_weights())

    def _update_target_nets(self):
        tvars = self.target_softq1.trainable_variables + self.target_softq2.trainable_variables
        mvars = self.softq1.trainable_variables + self.softq2.trainable_variables
        [tvar.assign(self.polyak * tvar + (1. - self.polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]

