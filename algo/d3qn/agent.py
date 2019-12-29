import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.tf_utils import n_step_target
from utility.losses import huber_loss
from utility.schedule import PiecewiseSchedule
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
        if hasattr(self, 'schedule_lr') and self.schedule_lr:
            self.lr_scheduler = PiecewiseSchedule(
                [(5e5, self.learning_rate), (2e6, 5e-5)], outside_value=5e-5)
            self.learning_rate = tf.Variable(self.learning_rate, trainable=False)
        else:
            self.schedule_lr = False
        self.optimizer = Optimizer(learning_rate=self.learning_rate,
                                    epsilon=self.epsilon)
        self.ckpt_models['optimizer'] = self.optimizer

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

    def learn_log(self, step):
        """ Update the network and do some bookkeeping
        step: the learning step
        """
        if self.schedule_lr:
            self.learning_rate.assign(self.lr_scheduler.value(step))

        data = self.dataset.sample()
        if self.dataset.buffer_type() != 'uniform':
            saved_indices = data['saved_indices']
            del data['saved_indices']
        q, loss, priority = self.learn(**data)
        if step % self.target_update_freq == 0:
            self._sync_target_nets()
        if self.dataset.buffer_type() != 'uniform':
            self.dataset.update_priorities(priority.numpy(), saved_indices.numpy())
        self.store(
            q=q.numpy(),
            loss=loss.numpy(),
            priority=priority.numpy(), 
        )

    @tf.function
    def _learn(self, IS_ratio, state, action, reward, next_state, done, steps):
        with tf.name_scope('q_train'):
            q, loss, priority, grads = self._compute_grads(
                state, action, next_state, reward, 
                done, steps, IS_ratio)
            if hasattr(self, 'clip_norm'):
                grads, norm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.optimizer.apply_gradients(
                zip(grads, self.q1.trainable_variables))

        return q, loss, priority

    def _compute_grads(self, state, action, next_state, reward, done, steps, IS_ratio):
        with tf.GradientTape() as tape:
            q = self.q1.train_value(state, action)
            next_action = self.q1.train_action(next_state)
            next_q = self.target_q1.train_det_value(next_state, next_action)

            loss_fn = tf.square if self.loss_type == 'mse' else huber_loss
            with tf.name_scope('q_loss'):
                target_q = n_step_target(reward, done, next_q, self.gamma, steps)
                error = tf.abs(target_q - q, name='q_error')
                
                loss = tf.reduce_mean(IS_ratio * loss_fn(error))
                
        priority = self._compute_priority(error)
        with tf.name_scope('q_grads'):
            grads = tape.gradient(loss, self.q1.trainable_variables)

        return q, loss, priority, grads

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        with tf.name_scope('priority'):
            priority += self.per_epsilon
            priority **= self.per_alpha
        
        return priority

    def _sync_target_nets(self):
        with Timer(f'sync', 10):
            self.target_q1.set_weights(self.q1.get_weights())
