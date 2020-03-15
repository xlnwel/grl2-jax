import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target
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
        self._ckpt_models['optimizer'] = self.optimizer

        self._action_dim = env.action_dim

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = [
            ([1], tf.float32, 'IS_ratio'),
            (env.obs_shape, tf.float32, 'obs'),
            ([env.action_dim], tf.float32, 'action'),
            ([1], tf.float32, 'reward'),
            (env.obs_shape, tf.float32, 'next_obs'),
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
        if not self.dataset.buffer_type().endswith('uniform'):
            saved_indices = data['saved_indices']
            del data['saved_indices']
        terms = self.learn(**data)
        if step % self.target_update_freq == 0:
            self._sync_target_nets()
        if not self.dataset.buffer_type().endswith('uniform'):
            self.dataset.update_priorities(terms['priority'].numpy(), saved_indices.numpy())
        self.store(**terms)

    @tf.function
    def _learn(self, **kwargs):
        with tf.name_scope('q_train'):
            grads, terms = self._compute_grads(**kwargs)
            if hasattr(self, 'clip_norm'):
                grads, norm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.optimizer.apply_gradients(
                zip(grads, self.q1.trainable_variables))

        return terms

    def _compute_grads(self, obs, action, next_obs, reward, done, steps, IS_ratio):
        action = tf.one_hot(action, self._action_dim)
        with tf.GradientTape() as tape:
            q = self.q1.train_value(obs, action)
            next_action = self.q1.train_action(next_obs)
            next_q = self.target_q1.train_det_value(next_obs, next_action)

            loss_fn = tf.square if self.loss_type == 'mse' else huber_loss
            with tf.name_scope('q_loss'):
                target_q = n_step_target(reward, done, next_q, self.gamma, steps)
                error = target_q - q
                
                loss = tf.reduce_mean(IS_ratio * loss_fn(error))
        
        if self.dataset.buffer_type().endswith('uniform'):
            priority = 1
        else:
            priority = self._compute_priority(tf.abs(error))
        
        with tf.name_scope('q_grads'):
            grads = tape.gradient(loss, self.q1.trainable_variables)

        return grads, dict(
            q=q, 
            loss=loss, 
            priority=priority,
        )

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        with tf.name_scope('priority'):
            priority += self.per_epsilon
            priority **= self.per_alpha
        
        return priority

    def _sync_target_nets(self):
        with Timer(f'sync', 10):
            self.target_q1.set_weights(self.q1.get_weights())
