import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target, transformed_n_step_target
from utility.schedule import PiecewiseSchedule, TFPiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from core.optimizers import Optimizer


class Agent(BaseAgent):
    @agent_config
    def __init__(self, 
                *,
                dataset,
                env):
        # dataset for input pipline optimization
        self.dataset = dataset
        self._is_per = not self.dataset.buffer_type().endswith('uniform')

        # learning rate schedule
        if getattr(self, '_schedule_lr', False):
            self._actor_sched = PiecewiseSchedule(
                [(2e5, self._actor_lr), (1e6, 1e-5)])
            self._q_sched = PiecewiseSchedule(
                [(2e5, self._q_lr), (1e6, 1e-5)])
            self._actor_lr = tf.Variable(self._actor_lr, trainable=False)
            self._q_lr = tf.Variable(self._q_lr, trainable=False)
            self.lr_pairs = [
                (self._actor_lr, self._actor_sched), (self._q_lr, self._q_sched)]

        # optimizer
        clip_norm = getattr(self, 'clip_norm', None)
        self._actor_opt = Optimizer(
            'adam', self.actor, self._actor_lr, 
            clip_norm=clip_norm)
        self._q_opt = Optimizer(
            'adam', [self.q1, self.q2], self._q_lr, 
            clip_norm=clip_norm)
        self._ckpt_models['_actor_opt'] = self._actor_opt
        self._ckpt_models['q_opt'] = self._q_opt

        if isinstance(self.temperature, float):
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            if getattr(self, '_schedule_lr', False):
                self._temp_sched = PiecewiseSchedule(
                    [(5e5, self._temp_lr), (1e6, 1e-5)])
                self._temp_lr = tf.Variable(self._temp_lr, trainable=False)
                self.lr_pairs.append((self._temp_lr, self._temp_sched))
            self._temp_opt = Optimizer(
                'adam', self.temperature, self._temp_lr, 
                clip_norm=clip_norm)
            self._ckpt_models['temp_opt'] = self._temp_opt

        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            state=(env.state_shape, tf.float32, 'state'),
            action=(env.action_shape, env.action_dtype, 'action'),
            reward=((), tf.float32, 'reward'),
            next_state=(env.state_shape, tf.float32, 'next_state'),
            done=((), tf.float32, 'done'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if 'steps'  in self.dataset.data_format:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs)

        self._sync_target_nets()

    def action(self, state, deterministic=False, epsilon=0):
        return self.actor.action(state, deterministic=deterministic, epsilon=epsilon)

    def learn_log(self, step=None):
        if step:
            self.global_steps.assign(step)
        if self._schedule_lr:
            [lr.assign(sched.value(self.global_steps.numpy())) for lr, sched in self.lr_pairs]
        with TBTimer(f'{self._model_name} sample', 10000, to_log=self.TIMER):
            data = self.dataset.sample()
        if self._is_per:
            saved_idxes = data['saved_idxes']
            del data['saved_idxes']

        with TBTimer(f'{self._model_name} learn', 10000, to_log=self.TIMER):
            terms = self.learn(**data)

        for k, v in terms.items():
            terms[k] = v.numpy()
            
        self._update_target_nets()

        if self._schedule_lr:
            terms['_actor_lr'] = self._actor_lr.numpy()
            terms['_q_lr'] = self._q_lr.numpy()
            if not isinstance(self.temperature, (float, tf.Variable)):
                terms['_temp_lr'] = self._temp_lr.numpy()
            
        if self._is_per:
            self.dataset.update_priorities(terms['priority'], saved_idxes.numpy())
        self.store(**terms)

    @tf.function
    def _learn(self, state, action, reward, next_state, done, IS_ratio=1, steps=1):
        target_entropy = getattr(self, 'target_entropy', -self._action_dim)
        if self._is_action_discrete:
            old_action = tf.one_hot(action, self._action_dim)
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
                
                target_q = target_fn(reward, done, nth_value, self._gamma, steps)
                q1_error = target_q - q1
                q2_error = target_q - q2

                tf.debugging.assert_shapes([(q1_error, (None,)), (q2_error, (None,))])

                q1_loss = .5 * tf.reduce_mean(IS_ratio * q1_error**2)
                q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
                q_loss = q1_loss + q2_loss

        if self._is_per:
            priority = self._compute_priority((tf.abs(q1_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority
            
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)
        terms['q_norm'] = self._q_opt(tape, q_loss)
        if not isinstance(self.temperature, (float, tf.Variable)):
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)
            
        terms.update(dict(
            actor_loss=actor_loss,
            q1=q1, 
            q2=q2,
            target_q=target_q,
            q1_loss=q1_loss, 
            q2_loss=q2_loss,
            q_loss=q_loss, 
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        with tf.name_scope('priority'):
            priority += self._per_epsilon
            priority **= self._per_alpha
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
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]
