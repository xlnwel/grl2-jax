import functools
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from utility.display import pwc
from utility.utils import AttrDict, Every
from utility.rl_utils import lambda_return, n_step_target
from utility.tf_utils import static_scan
from utility.schedule import PiecewiseSchedule, TFPiecewiseSchedule
from utility.graph import video_summary
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, display_model_var_info
from core.optimizer import Optimizer
from algo.dreamer1.nn import RSSMState


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        # dataset for input pipline optimization
        self.dataset = dataset
        self._dtype = global_policy().compute_dtype

        # optimizer
        dynamics_models = [self.encoder, self.rssm, self.decoder, self.reward]
        if hasattr(self, 'discount'):
            dynamics_models.append(self.discount)

        DreamerOpt = functools.partial(
            Optimizer,
            name=self._optimizer, 
            weight_decay=self._weight_decay, 
            clip_norm=self._clip_norm,
        )
        self._model_opt = DreamerOpt(models=dynamics_models, lr=self._model_lr)
        self._actor_opt = DreamerOpt(models=self.actor, lr=self._actor_lr)
        self._value_opt = DreamerOpt(models=[self.q1, self.q2], lr=self._value_lr)

        self._ckpt_models['model_opt'] = self._model_opt
        self._ckpt_models['actor_opt'] = self._actor_opt
        self._ckpt_models['value_opt'] = self._value_opt

        if isinstance(self.temperature, float):
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            self._temp_opt = DreamerOpt(models=self.temperature, lr=self._temp_lr)
            self._ckpt_models['temp_opt'] = self._temp_opt

        self._state = None
        self._prev_action = None

        self._obs_shape = env.obs_shape
        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        self._to_log_images = Every(self.LOG_INTERVAL)

        # time dimension must be explicitly specified here
        # otherwise, InaccessibleTensorError arises when expanding rssm
        TensorSpecs = dict(
            obs=((self._batch_len, *self._obs_shape), self._dtype, 'obs'),
            action=((self._batch_len, self._action_dim), self._dtype, 'action'),
            reward=((self._batch_len,), self._dtype, 'reward'),
            discount=((self._batch_len,), self._dtype, 'discount'),
            log_images=(None, tf.bool, 'log_images')
        )
        if self._store_state:
            state_size = self.rssm.state_size
            TensorSpecs['state'] = (RSSMState(
               *[((sz, ), self._dtype, name) for name, sz in zip(RSSMState._fields, state_size)]
            ))

        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

        self._sync_target_nets()

    def reset_states(self, state=None, prev_action=None):
        self._state = state
        self._prev_action = prev_action

    def retrieve_states(self):
        return self._state, self._prev_action

    def __call__(self, obs, reset=np.zeros(1), deterministic=False):
        if len(obs.shape) % 2 != 0:
            has_expanded = True
            obs = np.expand_dims(obs, 0)
        else:
            has_expanded = False
        if self._state is None and self._prev_action is None:
            self._state = self.rssm.get_initial_state(batch_size=tf.shape(obs)[0])
            self._prev_action = tf.zeros(
                (tf.shape(obs)[0], self._action_dim), self._dtype)
        if np.any(reset):
            mask = tf.cast(1. - reset, self._dtype)[:, None]
            self._state = tf.nest.map_structure(lambda x: x * mask, self._state)
            self._prev_action = self._prev_action * mask
        action, self._state = self.action(
            obs, self._state, self._prev_action, deterministic)
        
        self._prev_action = tf.one_hot(action, self._action_dim, dtype=self._dtype) \
            if self._is_action_discrete else action
        
        action = np.squeeze(action.numpy()) if has_expanded else action.numpy()
        if self._store_state:
            return action, tf.nest.map_structure(lambda x: x.numpy(), self._state)
        else:
            return action
        
    @tf.function
    def action(self, obs, state, prev_action, deterministic=False):
        if obs.dtype == np.uint8:
            obs = tf.cast(obs, self._dtype) / 255. - .5

        obs = tf.expand_dims(obs, 1)
        embed = self.encoder(obs)
        embed = tf.squeeze(embed, 1)
        state = self.rssm.post_step(state, prev_action, embed)
        feature = self.rssm.get_feat(state)
        action = self.actor.action(feature, deterministic, self._act_epsilon)
            
        return action, state

    def learn_log(self, step):
        self.global_steps.assign(step)
        for i in range(self.N_UPDATES):
            data = self.dataset.sample()
            log_images = tf.convert_to_tensor(
                self._log_images and i == 0 and self._to_log_images(step), 
                tf.bool)
            terms = self.learn(**data, log_images=log_images)
            terms = {k: v.numpy() for k, v in terms.items()}
            self.store(**terms)
            self._update_target_nets()

    @tf.function
    def _learn(self, obs, action, reward, discount, log_images, state=None):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(obs)
            if self._burn_in:
                bl = self._burn_in_len
                sl = self._batch_len - self._burn_in_len
                burn_in_embed, embed = tf.split(embed, [bl, sl], 1)
                burn_in_action, action = tf.split(action, [bl, sl], 1)
                state, _ = self.rssm.observe(burn_in_embed, burn_in_action, state)
                state = tf.nest.pack_sequence_as(state, 
                    tf.nest.map_structure(lambda x: tf.stop_gradient(x[:, -1]), state))
                
                _, obs = tf.split(obs, [bl, sl], 1)
                _, reward = tf.split(reward, [bl, sl], 1)
                _, discount = tf.split(discount, [bl, sl], 1)
            post, prior = self.rssm.observe(embed, action, state)
            feature = self.rssm.get_feat(post)
            obs_pred = self.decoder(feature)
            reward_pred = self.reward(feature)
            likelihoods = AttrDict()
            likelihoods.obs_loss = -tf.reduce_mean(obs_pred.log_prob(obs))
            likelihoods.reward_loss = -tf.reduce_mean(reward_pred.log_prob(reward))
            if hasattr(self, 'discount'):
                disc_pred = self.discount(feature)
                disc_target = self._gamma * discount
                likelihoods.disc_loss = -(self._discount_scale 
                    * tf.reduce_mean(disc_pred.log_prob(disc_target)))
            prior_dist = self.rssm.get_dist(prior.mean, prior.std)
            post_dist = self.rssm.get_dist(post.mean, post.std)
            kl = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            kl = tf.maximum(kl, self._free_nats)
            model_loss = self._kl_scale * kl + sum(likelihoods.values())

        target_entropy = getattr(self, 'target_entropy', -self._action_dim)
        curr_feat = feature[:, :-1]
        nth_feat = feature[:, 1:]
        action = action[:, :-1]
        reward = reward[:, :-1]
        discount = discount[:, :-1]
        with tf.GradientTape(persistent=True) as ac_tape:
            new_action, logpi, terms = self.actor.train_step(curr_feat)
            q1_with_actor = self.q1(curr_feat, new_action)
            q2_with_actor = self.q2(curr_feat, new_action)
            q_with_actor = tf.minimum(q1_with_actor, q2_with_actor)

            nth_action, nth_logpi, _ = self.actor.train_step(nth_feat)
            nth_q1_with_actor = self.target_q1(nth_feat, nth_action)
            nth_q2_with_actor = self.target_q2(nth_feat, nth_action)
            nth_q_with_actor = tf.minimum(nth_q1_with_actor, nth_q2_with_actor)
            
            if isinstance(self.temperature, (float, tf.Variable)):
                temp = nth_temp = self.temperature
            else:
                log_temp, temp = self.temperature(curr_feat, action)
                _, nth_temp = self.temperature(nth_feat, nth_action)
                temp_loss = -tf.reduce_mean(log_temp 
                    * tf.stop_gradient(logpi + target_entropy))
                terms['temp'] = temp
                terms['temp_loss'] = temp_loss

            q1 = self.q1(curr_feat, action)
            q2 = self.q2(curr_feat, action)

            tf.debugging.assert_shapes(
                [(q1, (None, self._batch_len - 1)), 
                (q2, (None, self._batch_len - 1)), 
                (logpi, (None, self._batch_len - 1)), 
                (q_with_actor, (None, self._batch_len - 1)), 
                (nth_q_with_actor, (None, self._batch_len - 1))])
            
            actor_loss = tf.reduce_mean(tf.stop_gradient(temp) * logpi - q_with_actor)

            nth_value = nth_q_with_actor - nth_temp * nth_logpi
            target_q = n_step_target(reward, nth_value, discount, self._gamma)
            q1_error = target_q - q1
            q2_error = target_q - q2

            tf.debugging.assert_shapes(
                [(q1_error, (None, self._batch_len - 1)), 
                (q2_error, (None, self._batch_len - 1))])

            q1_loss = .5 * tf.reduce_mean(q1_error**2)
            q2_loss = .5 * tf.reduce_mean(q2_error**2)
            value_loss = q1_loss + q2_loss

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(ac_tape, actor_loss)
        value_norm = self._value_opt(ac_tape, value_loss)
        if not isinstance(self.temperature, (float, tf.Variable)):
            terms['temp_norm'] = self._temp_opt(ac_tape, temp_loss)
        
        terms.update(dict(
            prior_entropy=prior_dist.entropy(),
            post_entropy=post_dist.entropy(),
            kl=kl,
            q1=q1,
            q2=q2,
            logpi=logpi,
            **likelihoods,
            model_loss=model_loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            model_norm=model_norm,
            actor_norm=actor_norm,
            value_norm=value_norm,
        ))

        if log_images:
            self._image_summaries(obs, action, embed, obs_pred)
    
        return terms

    def _image_summaries(self, obs, action, embed, image_pred):
        truth = obs[:6] + 0.5
        recon = image_pred.mode()[:6] + .5
        error = (recon - truth + 1) / 2
        openl = tf.concat([truth, recon, error], 2)
        self.graph_summary(video_summary, 'dreamer/comp', openl)

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_q1.variables + self.target_q2.variables
        mvars = self.q1.variables + self.q2.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]

    @tf.function
    def _update_target_nets(self):
        tvars = self.target_q1.trainable_variables + self.target_q2.trainable_variables
        mvars = self.q1.trainable_variables + self.q2.trainable_variables
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(tvars, mvars)]
