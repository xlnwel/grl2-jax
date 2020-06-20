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


def center_crop_image(image, output_size):
    h, w = image.shape[:2]
    new_h, new_w = output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[top:top + new_h, left:left + new_w]
    assert image.shape[:2] == output_size
    return image

class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._is_per = dataset.name().endswith('per')
        self.dataset = dataset

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        self._q_opt = Optimizer(self._optimizer, [self.encoder, self.q1, self.q2], self._q_lr)
        self._curl_opt = Optimizer(self._optimizer, [self.encoder, self.curl], self._curl_lr)
        if not isinstance(self.temperature, (float, tf.Variable)):
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr, beta_1=.5)

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
            update_actor=(None, tf.bool, 'update_actor')
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if 'steps'  in self.dataset.data_format:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs, 
                            batch_size=self._batch_size, 
                            print_terminal_info=True)

        self._sync_target_nets()

    def __call__(self, obs, deterministic=False, **kwargs):
        obs = np.array(obs)
        obs = center_crop_image(obs, self._obs_shape[:2])
        if len(obs.shape) % 2 == 1:
            obs = np.expand_dims(obs, 0)
        action = self.action(obs, deterministic)
        return np.squeeze(action.numpy(), 0)

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

        terms = {k: v.numpy() for k, v in terms.items()}

        if self._is_per:
            self.dataset.update_priorities(terms['priority'], idxes)
        self.store(**terms)
        
        return 1

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, obs_pos, steps=1, IS_ratio=1):
        next_x_actor = self.encoder.cnn(next_obs)
        next_action, next_logpi, _ = self.actor.train_step(next_x_actor)

        next_x_value = self.target_encoder.cnn(next_obs)
        next_z_value = self.target_encoder.mlp(next_x_value)
        next_q1_with_actor = self.target_q1(next_z_value, next_action)
        next_q2_with_actor = self.target_q2(next_z_value, next_action)
        next_q_with_actor = tf.minimum(next_q1_with_actor, next_q2_with_actor)
        temp = self.temperature()
        next_value = next_q_with_actor - temp * next_logpi
        target_q = n_step_target(reward, next_value, discount, self._gamma, steps)
        target_q = tf.stop_gradient(target_q)
        
        terms = {}
        with tf.GradientTape(persistent=True) as tape:
            x = self.encoder.cnn(obs)
            z = self.encoder.mlp(x)

            q1 = self.q1(z, action)
            q2 = self.q2(z, action)
            q1_error = target_q - q1
            q2_error = target_q - q2
            q1_loss = .5 * tf.reduce_mean(IS_ratio * q1_error**2)
            q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
            q_loss = q1_loss + q2_loss

            x_pos = self.target_encoder.cnn(obs_pos)
            z_pos = self.target_encoder.mlp(x_pos)
            logits = self.curl(z, z_pos)
            tf.debugging.assert_shapes([[logits, (self._batch_size, self._batch_size)]])
            labels = tf.range(logits.shape[0])
            curl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            curl_loss = tf.reduce_mean(curl_loss)

        terms['q_norm'] = self._q_opt(tape, q_loss)
        terms['curl_norm'] = self._curl_opt(tape, curl_loss)

        if self._is_per:
            priority = self._compute_priority((tf.abs(q1_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority
        
        with tf.GradientTape(persistent=True) as actor_tape:
            temp = self.temperature()
            new_action, logpi, _ = self.actor.train_step(x)
            temp_loss = -tf.reduce_mean(IS_ratio * temp 
                * tf.stop_gradient(logpi + self._target_entropy))
            q1_with_actor = self.q1(z, new_action)
            q2_with_actor = self.q2(z, new_action)
            q_with_actor = tf.minimum(q1_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(IS_ratio * (tf.stop_gradient(temp) * logpi - q_with_actor))

        terms['actor_norm'] = self._actor_opt(actor_tape, actor_loss)
        terms['temp_norm'] = self._temp_opt(actor_tape, temp_loss)
    
        terms.update(dict(
            temp=temp,
            q1=q1, 
            q2=q2,
            target_q=target_q,
            q1_loss=q1_loss, 
            q2_loss=q2_loss,
            q_loss=q_loss, 
            curl_loss=curl_loss,
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
        tvars = self.target_q1.variables + self.target_q2.variables + self.target_encoder.variables
        mvars = self.q1.variables + self.q2.variables + self.encoder.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]

    @tf.function
    def _update_target_qs(self):
        tvars = self.target_q1.variables + self.target_q2.variables
        mvars = self.q1.variables + self.q2.variables
        [tvar.assign(self._q_polyak * tvar + (1. - self._q_polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]

    @tf.function
    def _update_target_encoder(self):
        [tvar.assign(self._encoder_polyak * tvar + (1. - self._encoder_polyak) * mvar) 
            for tvar, mvar in zip(self.target_encoder.variables, self.encoder.variables)]
