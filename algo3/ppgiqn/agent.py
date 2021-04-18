import logging
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.schedule import TFPiecewiseSchedule
from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss, quantile_regression_loss
from core.tf_config import build
from core.optimizer import Optimizer
from core.decorator import override
from algo.ppg.agent import Agent as PPGAgent


logger = logging.getLogger(__name__)

class Agent(PPGAgent):
    """ Initialization """
    @override(PPGAgent)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._actor_lr, list)
            assert isinstance(self._value_lr, list)
            assert isinstance(self._aux_lr, list)
            self._actor_lr = TFPiecewiseSchedule(self._actor_lr)
            self._value_lr = TFPiecewiseSchedule(self._value_lr)
            self._aux_lr = TFPiecewiseSchedule(self._aux_lr)

        actor_models = [self.encoder, self.actor]
        if hasattr(self, 'rnn'):
            actor_models.append(self.rnn)
        self._actor_opt = Optimizer(
            self._optimizer, actor_models, self._actor_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

        value_models = [self.value, self.quantile]
        if hasattr(self, 'value_encoder'):
            value_models.append(self.value_encoder)
        if hasattr(self, 'value_rnn'):
            value_models.append(self.value_rnn)
        self._value_opt = Optimizer(
            self._optimizer, value_models, self._value_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)
        
        aux_models = list(self.model.values())
        self._aux_opt = Optimizer(
            self._optimizer, aux_models, self._aux_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

    @override(PPGAgent)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            value=((self.N,), tf.float32, 'value'),
            traj_ret=((self.N,), tf.float32, 'traj_ret'),
            advantage=((self.N,), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            logits=((env.action_dim,), tf.float32, 'logits'),
            value=((self.N,), tf.float32, 'value'),
            traj_ret=((self.N,), tf.float32, 'traj_ret'),
        )
        self.aux_learn = build(self._aux_learn, TensorSpecs, batch_size=self._aux_batch_size)

    def _process_input(self, env_output, evaluation):
        obs, kwargs = super()._process_input(env_output, evaluation)
        kwargs['tau_hat'] = self._tau_hat
        return obs, kwargs

    def before_run(self, env):
        self._tau_hat = self.quantile.sample_tau(env.n_envs)
        self._aux_tau_hat = self.quantile.sample_tau(self._aux_batch_size)

    def compute_value(self, obs=None):
        # be sure you normalize obs first if obs normalization is required
        obs = obs or self._last_obs
        return self.model.compute_value(obs, self._tau_hat).numpy()
    
    def compute_aux_data(self, obs):
        out = self.model.compute_aux_data(obs, self._aux_tau_hat)
        out = tf.nest.map_structure(lambda x: x.numpy(), out)
        return out

    @tf.function
    def _learn(self, obs, action, value, traj_ret, advantage, logpi, state=None, mask=None):
        terms = {}
        advantage = tf.reduce_mean(advantage, axis=-1)
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)    
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            actor_loss = (policy_loss - self._entropy_coef * entropy)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        with tf.GradientTape() as tape:
            x_value = self.value_encoder(obs)
            tau_hat, qt_embed = self.quantile(x_value, self.N)
            x_value = tf.expand_dims(x_value, 1)
            value = self.value(x_value, qt_embed)
            value_ext = tf.expand_dims(value, axis=-1)
            traj_ret_ext = tf.expand_dims(traj_ret, axis=1)
            value_loss = quantile_regression_loss(
                value_ext, traj_ret_ext, tau_hat, kappa=self.KAPPA)
            value_loss = tf.reduce_mean(value_loss)
        terms['value_norm'] = self._value_opt(tape, value_loss)

        terms.update(dict(
            value=value,
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value)
        ))

        return terms
    
    @tf.function
    def learn_policy(self, obs, action, advantage, logpi, state=None, mask=None):
        terms = {}
        advantage = tf.reduce_mean(advantage, axis=-1)
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)    
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            actor_loss = (policy_loss - self._entropy_coef * entropy)
        actor_norm = self._actor_opt(tape, actor_loss)
        terms.update(dict(
            ratio=tf.exp(log_ratio),
            entropy=entropy,
            kl=kl,
            p_clip_frac=p_clip_frac,
            ppo_loss = policy_loss,
            actor_loss = actor_loss,
            actor_norm=actor_norm
        ))
        return terms
    
    @tf.function
    def learn_value(self, obs, value, traj_ret, state, mask=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x_value = self.value_encoder(obs)
            if state is not None:
                x_value, state = self.value_rnn(x_value, state, mask=mask)
            value = self.value(x_value)
            value_loss = self._compute_value_loss(value, traj_ret, old_value, terms)
        value_norm = self._value_opt(tape, value_loss)
        terms.update(dict(
            value_norm=value_norm,
            value_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value)
        ))

    @tf.function
    def _aux_learn(self, obs, logits, value, traj_ret, mask=None, state=None):
        terms = {}
        traj_ret_ext = tf.expand_dims(traj_ret, axis=1)
        traj_ret = tf.reduce_mean(traj_ret, axis=-1)
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)
            act_dist = self.actor(x)
            old_dist = tfd.Categorical(logits=logits)
            kl = tf.reduce_mean(old_dist.kl_divergence(act_dist))
            actor_loss = bc_loss = self._bc_coef * kl
            aux_value = self.aux_value(x)
            aux_loss = .5 * tf.reduce_mean((aux_value - traj_ret)**2)
            terms['bc_loss'] = bc_loss
            terms['aux_loss'] = aux_loss
            actor_loss = aux_loss + bc_loss
            
            x_value = self.value_encoder(obs)
            tau_hat, qt_embed = self.quantile(x_value, self.N)
            x_value = tf.expand_dims(x_value, 1)
            value = self.value(x_value, qt_embed)
            value_ext = tf.expand_dims(value, axis=-1)
            value_loss = quantile_regression_loss(
                value_ext, traj_ret_ext, tau_hat, kappa=self.KAPPA)
            value_loss = tf.reduce_mean(value_loss)
            loss = actor_loss + value_loss

        terms['actor_norm'] = self._aux_opt(tape, loss)

        terms = dict(
            value=value, 
            kl=kl, 
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value)
        )

        return terms
