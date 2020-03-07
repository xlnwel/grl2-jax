import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target, transformed_n_step_target
from utility.schedule import PiecewiseSchedule, TFPiecewiseSchedule
from utility.timer import TBTimer
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
        # dataset for input pipline optimization
        self.dataset = dataset
        self.is_per = not self.dataset.buffer_type().endswith('uniform')

        # learning rate schedule
        if getattr(self, 'schedule_lr', False):
            self.actor_schedule = PiecewiseSchedule(
                [(2e5, self.actor_lr), (1e6, 1e-5)])
            self.q_schedule = PiecewiseSchedule(
                [(2e5, self.q_lr), (1e6, 1e-5)])
            self.actor_lr = tf.Variable(self.actor_lr, trainable=False)
            self.q_lr = tf.Variable(self.q_lr, trainable=False)

        # optimizer
        if self.optimizer.lower() == 'adam':
            Optimizer = tf.keras.optimizers.Adam
        elif self.optimizer.lower() == 'rmsprop':
            Optimizer = tf.keras.optimizers.RMSprop
        else:
            raise NotImplementedError()
        self.actor_opt = Optimizer(learning_rate=self.actor_lr,
                                    epsilon=self.epsilon)
        self.q_opt = Optimizer(learning_rate=self.q_lr,
                                epsilon=self.epsilon)
        self.ckpt_models['actor_opt'] = self.actor_opt
        self.ckpt_models['q_opt'] = self.q_opt

        if isinstance(self.temperature, float):
            # if env.name == 'BipedalWalkerHardcore-v2':
            #     self.temp_schedule = PiecewiseSchedule(
            #         [(5e5, self.temperature), (1e7, 0)])
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            if getattr(self, 'schedule_lr', False):
                self.temp_schedule = PiecewiseSchedule(
                    [(5e5, self.temp_lr), (1e6, 1e-5)])
                self.temp_lr = tf.Variable(self.temp_lr, trainable=False)
            self.temp_opt = Optimizer(learning_rate=self.temp_lr,
                                    epsilon=self.epsilon)
            self.ckpt_models['temp_opt'] = self.temp_opt

        self.action_dim = env.action_dim
        self.is_action_discrete = env.is_action_discrete

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            IS_ratio=((), tf.float32, 'IS_ratio'),
            state=(env.state_shape, tf.float32, 'state'),
            action=([env.action_dim], tf.float32, 'action'),
            reward=((), tf.float32, 'reward'),
            next_state=(env.state_shape, tf.float32, 'next_state'),
            done=((), tf.float32, 'done'),
            steps=((), tf.float32, 'steps'),
        )
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def action(self, state, deterministic=False, epsilon=0):
        return self.actor.action(state, deterministic=deterministic, epsilon=epsilon)

    def learn_log(self, step=None):
        if step:
            self.global_steps.assign(step)
        if self.schedule_lr:
            self.actor_lr.assign(self.actor_schedule.value(self.global_steps.numpy()))
            self.q_lr.assign(self.q_schedule.value(self.global_steps.numpy()))
            if not isinstance(self.temperature, (float, tf.Variable)):
                self.temp_lr.assign(self.temp_schedule.value(self.global_steps.numpy()))
        if hasattr(self, 'temp_schedule'):
            self.temperature.assign(self.temp_schedule.value(self.global_steps.numpy()))
        with TBTimer(f'{self.model_name} sample', 10000, to_log=self.timer):
            data = self.dataset.sample()
        if self.is_per:
            saved_idxes = data['saved_idxes']
            del data['saved_idxes']

        with TBTimer(f'{self.model_name} learn', 10000, to_log=self.timer):
            terms = self.learn(**data)

        for k, v in terms.items():
            terms[k] = v.numpy()
            
        self._update_target_nets()

        if self.schedule_lr:
            terms['actor_lr'] = self.actor_lr.numpy()
            terms['q_lr'] = self.q_lr.numpy()
            if not isinstance(self.temperature, (float, tf.Variable)):
                terms['temp_lr'] = self.temp_lr.numpy()
            
        if self.is_per:
            self.dataset.update_priorities(terms['priority'], saved_idxes.numpy())
        self.store(**terms)

    @tf.function
    def _learn(self, **kwargs):
        with tf.name_scope('grads'):
            grads, terms = self._compute_grads(**kwargs)

        with tf.name_scope('actor_update'):
            actor_grads = grads['actor_grads']
            if getattr(self, 'clip_norm', None) is not None:
                actor_grads, actor_norm = tf.clip_by_global_norm(actor_grads, self.clip_norm)
                terms['actor_norm'] = actor_norm
            self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        with tf.name_scope('q_update'):
            q_grads = grads['q_grads']
            if getattr(self, 'clip_norm', None) is not None:
                q_grads, q_norm = tf.clip_by_global_norm(q_grads, self.clip_norm)
                terms['q_norm'] = q_norm
            self.q_opt.apply_gradients(
                zip(q_grads, self.q1.trainable_variables + self.q2.trainable_variables))

        if not isinstance(self.temperature, (float, tf.Variable)):
            with tf.name_scope('temp_update'):
                temp_grads = grads['temp_grads']
                if getattr(self, 'clip_norm', None) is not None:
                    temp_grads, temp_norm = tf.clip_by_global_norm(temp_grads, self.clip_norm)
                    terms['temp_norm'] = temp_norm
                self.temp_opt.apply_gradients(zip(temp_grads, self.temperature.trainable_variables))

        return terms

    def _compute_grads(self, IS_ratio, state, action, reward, next_state, done, steps=1):
        target_entropy = getattr(self, 'target_entropy', -self.action_dim)
        if self.is_action_discrete:
            old_action = tf.one_hot(action, self.action_dim)
        else:
            old_action = action
        target_fn = (transformed_n_step_target if getattr(self, 'tbo', False) 
                    else n_step_target)
        with tf.GradientTape(persistent=True) as tape:
            action, logpi, terms = self.actor.train_step(state)
            q1_with_actor = self.q1.train_step(state, action)
            q2_with_actor = self.q2.train_step(state, action)
            q_with_actor = tf.minimum(q1_with_actor, q2_with_actor)

            next_action, next_logpi, _ = self.actor.train_step(next_state)
            next_q1_with_actor = self.target_q1.train_step(next_state, next_action)
            next_q2_with_actor = self.target_q2.train_step(next_state, next_action)
            next_q_with_actor = tf.minimum(next_q1_with_actor, next_q2_with_actor)
            
            if isinstance(self.temperature, (float, tf.Variable)):
                temp = next_temp = self.temperature
            else:
                log_temp, temp = self.temperature.train_step(state, action)
                _, next_temp = self.temperature.train_step(next_state, next_action)
                with tf.name_scope('temp_loss'):
                    temp_loss = -tf.reduce_mean(IS_ratio * log_temp * tf.stop_gradient(logpi + target_entropy))
                terms['temp'] = temp

            q1 = self.q1.train_step(state, old_action)
            q2 = self.q2.train_step(state, old_action)

            tf.debugging.assert_shapes(
                [(IS_ratio, (None,)),
                (q1, (None,)), 
                (q2, (None,)), 
                (logpi, (None,)), 
                (q_with_actor, (None,)), 
                (next_q_with_actor, (None,))])
            
            with tf.name_scope('actor_loss'):
                actor_loss = tf.reduce_mean(IS_ratio * tf.stop_gradient(temp) * logpi - q_with_actor)

            with tf.name_scope('q_loss'):
                nth_value = next_q_with_actor - next_temp * next_logpi

                tf.debugging.assert_shapes([(nth_value, (None,)), (reward, (None,)), (done, (None,)), (steps, (None,))])
                
                target_q = target_fn(reward, done, nth_value, self.gamma, steps)
                q1_error = target_q - q1
                q2_error = target_q - q2

                tf.debugging.assert_shapes([(q1_error, (None,)), (q2_error, (None,))])

                q1_loss = .5 * tf.reduce_mean(IS_ratio * q1_error**2)
                q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
                q_loss = q1_loss + q2_loss
            
        with tf.name_scope('actor_grads'):
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        with tf.name_scope('q_grads'):
            q_grads = tape.gradient(q_loss, 
                self.q1.trainable_variables + self.q2.trainable_variables)

        if self.is_per:
            priority = self._compute_priority((tf.abs(q1_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority
            
        grads = dict(
            actor_grads=actor_grads,
            q_grads=q_grads,
        )
        terms.update(dict(
            actor_loss=actor_loss,
            q1=q1, 
            q2=q2,
            target_q=target_q,
            q1_loss=q1_loss, 
            q2_loss=q2_loss,
            q_loss=q_loss, 
        ))
        if not isinstance(self.temperature, (float, tf.Variable)):
            with tf.name_scope('temp_grads'):
                temp_grads = tape.gradient(temp_loss, self.temperature.trainable_variables)
            grads['temp_grads'] = temp_grads
            terms['temp_loss'] = temp_loss

        return grads, terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        with tf.name_scope('priority'):
            priority += self.per_epsilon
            priority **= self.per_alpha
        tf.debugging.assert_greater(priority, 0.)
        return priority

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_q1.variables + self.target_q2.variables
        mvars = self.q1.variables + self.q2.variables
        assert len(tvars) == len(mvars)
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]

    @tf.function
    def _update_target_nets(self):
        tvars = self.target_q1.trainable_variables + self.target_q2.trainable_variables
        mvars = self.q1.trainable_variables + self.q2.trainable_variables
        assert len(tvars) == len(mvars)
        [tvar.assign(self.polyak * tvar + (1. - self.polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]
