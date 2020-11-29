import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.rl_utils import n_step_target
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

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr, return_grads=True)
        self._q_opt = Optimizer(self._optimizer, [self.encoder, self.q], self._q_lr, return_grads=True)
        # self._curl_opt = Optimizer(self._optimizer, [self.encoder, self.curl], self._curl_lr)
        if not isinstance(self.temperature, (float, tf.Variable)):
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr, beta_1=.5, return_grads=True)

        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete
        if not hasattr(self, '_target_entropy'):
            self._target_entropy = .98 * np.log(self._action_dim) \
                if self._is_action_discrete else -self._action_dim

        self._obs_shape = tuple(self._obs_shape) + env.obs_shape[-1:]
        TensorSpecs = dict(
            obs=(self._obs_shape, env.obs_dtype, 'obs'),
            action=((env.action_dim,), tf.float32, 'action'),
            reward=((), tf.float32, 'reward'),
            next_obs=(self._obs_shape, env.obs_dtype, 'next_obs'),
            discount=((), tf.float32, 'discount'),
            obs_pos=(self._obs_shape, env.obs_dtype, 'obs_pos'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if 'steps'  in self.dataset.data_format:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs, 
                            batch_size=self._batch_size, 
                            print_terminal_info=True)

        self._sync_target_nets()

    def __call__(self, obs, evaluation=False, **kwargs):
        return self.model.action(
            tf.convert_to_tensor(obs), 
            deterministic=evaluation, 
            epsilon=self._act_eps).numpy()

    @tf.function
    def action(self, obs, deterministic=False, **kwargs):
        x = self.encoder.cnn(obs)
        action = self.actor.action(x, deterministic=deterministic, epsilon=self._act_eps)
        return action

    @step_track
    def learn_log(self, step):
        data = self.dataset.sample()
        if self._is_per:
            idxes = data.pop('idxes').numpy()

        terms = self.learn(**data)
        if self.train_step % 2 == 0:
            self._update_target_qs()
            self._update_target_encoder()
        grads = terms.pop('grads')
        terms = {k: v.numpy() for k, v in terms.items()}

        if self._is_per:
            self.dataset.update_priorities(terms['priority'], idxes)
        self.store(**terms)

        if self.train_step % 1000 == 0:
            self.summary(data, grads)

        return 1

    @tf.function
    def summary(self, data, grads):
        self.histogram_summary({'reward': data['reward']})
        self.histogram_summary(grads)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, obs_pos, steps=1, IS_ratio=1):
        next_x_actor = self.encoder.cnn(next_obs)
        next_action, next_logpi, _ = self.actor.train_step(next_x_actor)

        next_x_value = self.target_encoder.cnn(next_obs)
        next_z_value = self.target_encoder.mlp(next_x_value)
        next_q_with_actor = self.target_q(next_z_value, next_action)
        temp = self.temperature()
        next_value = next_q_with_actor - temp * next_logpi
        target_q = n_step_target(reward, next_value, discount, self._gamma, steps)
        target_q = tf.stop_gradient(target_q)
        self.histogram_summary({
            'target_q': target_q, 
            'next_q_with_actor': next_q_with_actor,
            'next_logpi': next_logpi
        }, step=self._env_step)
        terms = {}
        with tf.GradientTape(persistent=True) as tape:
            x = self.encoder.cnn(obs)
            z = self.encoder.mlp(x)

            q = self.q(z, action)
            q_error = target_q - q
            q_loss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
            q_loss = q_loss

            # x_pos = self.target_encoder.cnn(obs_pos)
            # z_pos = self.target_encoder.mlp(x_pos)
            # logits = self.curl(z, z_pos)
            # tf.debugging.assert_shapes([[logits, (self._batch_size, self._batch_size)]])
            # labels = tf.range(logits.shape[0])
            # curl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=labels, logits=logits)
            # curl_loss = tf.reduce_mean(curl_loss)

        terms['q_norm'], q_grads = self._q_opt(tape, q_loss)
        # terms['curl_norm'] = self._curl_opt(tape, curl_loss)

        if self._is_per:
            priority = self._compute_priority((tf.abs(q_error)) / 2.)
            terms['priority'] = priority
        
        with tf.GradientTape(persistent=True) as actor_tape:
            x = self.encoder.cnn(obs)
            tf.stop_gradient(x)
            temp = self.temperature()
            new_action, logpi, actor_terms = self.actor.train_step(x)
            terms.update(actor_terms)
            temp_loss = -tf.reduce_mean(IS_ratio * temp 
                * tf.stop_gradient(logpi + self._target_entropy))
            q_with_actor = self.q(z, new_action)
            actor_loss = tf.reduce_mean(IS_ratio * (tf.stop_gradient(temp) * logpi - q_with_actor))

        terms['actor_norm'], act_grads = self._actor_opt(actor_tape, actor_loss)
        terms['temp_norm'], temp_grads = self._temp_opt(actor_tape, temp_loss)

        terms['grads'] = {**q_grads, **act_grads, **temp_grads}

        terms.update(dict(
            temp=temp,
            q=q, 
            target_q=target_q,
            q_loss=q_loss, 
            # curl_loss=curl_loss,
            actor_loss=actor_loss,
            temp_loss=temp_loss,
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
        tvars = self.target_q.variables + self.target_encoder.variables
        mvars = self.q.variables + self.encoder.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]

    @tf.function
    def _update_target_qs(self):
        tvars = self.target_q.variables
        mvars = self.q.variables
        [tvar.assign(self._q_polyak * tvar + (1. - self._q_polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]

    @tf.function
    def _update_target_encoder(self):
        [tvar.assign(self._encoder_polyak * tvar + (1. - self._encoder_polyak) * mvar) 
            for tvar, mvar in zip(self.target_encoder.variables, self.encoder.variables)]
