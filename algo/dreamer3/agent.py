import functools
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.utils import AttrDict, Every
from utility.rl_utils import retrace_lambda
from utility.tf_utils import static_scan, huber_loss
from utility.schedule import PiecewiseSchedule, TFPiecewiseSchedule
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.dreamer3.nn import RSSMState


def get_data_format(env, batch_size, sample_size=None, 
        store_state=False, state_size=None, dtype=tf.float32, **kwargs):
    data_format = dict(
        obs=((batch_size, sample_size, *env.obs_shape), dtype),
        prev_action=((batch_size, sample_size, *env.action_shape), dtype),
        reward=((batch_size, sample_size), dtype), 
        discount=((batch_size, sample_size), dtype),
        prev_logpi=((batch_size, sample_size), dtype)
    )
    if store_state:
        data_format.update({
            k: ((batch_size, v), dtype)
                for k, v in state_size._asdict().items()
        })
    return data_format

class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        # dataset for input pipline optimization
        self.dataset = dataset

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
        self._actor_opt = DreamerOpt(models=self.actor, lr=self._actor_lr, return_grads=True)
        self._q_opt = DreamerOpt(models=[self.q1, self.q2], lr=self._q_lr, return_grads=True)

        if isinstance(self.temperature, float):
            # convert to variable, useful for scheduling
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            if getattr(self, '_schedule_lr', False):
                self._temp_lr = TFPiecewiseSchedule(
                    [(5e5, self._temp_lr), (1e6, 1e-5)])
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

        self._state = None
        self._prev_action = None

        self._obs_shape = env.obs_shape
        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        self._to_log_images = Every(self.LOG_PERIOD)

        # time dimension must be explicitly specified here
        # otherwise, InaccessibleTensorError arises when expanding rssm
        TensorSpecs = dict(
            obs=((self._sample_size, *self._obs_shape), self._dtype, 'obs'),
            prev_action=((self._sample_size, self._action_dim), tf.float32, 'action'),
            reward=((self._sample_size,), tf.float32, 'reward'),
            discount=((self._sample_size,), tf.float32, 'discount'),
            prev_logpi=((self._sample_size,), tf.float32, 'logpi'),
            log_images=(None, tf.bool, 'log_images')
        )
        if self._store_state:
            state_size = self.rssm.state_size
            TensorSpecs['state'] = (RSSMState(
               *[((sz, ), tf.float32, name) 
               for name, sz in zip(RSSMState._fields, state_size)]
            ))

        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

        self._sync_target_nets()
        
    def reset_states(self, state=(None, None)):
        self._state, self._prev_action = state

    def get_states(self):
        return self._state, self._prev_action

    def __call__(self, obs, reset=np.zeros(1), deterministic=False, **kwargs):
        if len(obs.shape) % 2 != 0:
            has_expanded = True
            obs = np.expand_dims(obs, 0)
        else:
            has_expanded = False
        if self._state is None and self._prev_action is None:
            self._state = self.rssm.get_initial_state(batch_size=tf.shape(obs)[0])
            self._prev_action = tf.zeros(
                (tf.shape(obs)[0], self._action_dim), tf.float32)
        if np.any(reset):
            mask = tf.cast(1. - reset, tf.float32)[:, None]
            self._state = tf.nest.map_structure(lambda x: x * mask, self._state)
            self._prev_action = self._prev_action * mask
        prev_state = self._state
        if deterministic:
            action, self._state = self.action(
                obs, self._state, self._prev_action, deterministic)
        else:
            action, terms, self._state = self.action(
                obs, self._state, self._prev_action, deterministic)
        self._prev_action = tf.one_hot(action, self._action_dim, dtype=tf.float32) \
            if self._is_action_discrete else action
        
        action = np.squeeze(action.numpy())
        if deterministic:
            return action
        else:
            if self._store_state:
                terms.update(prev_state._asdict())
            terms = tf.nest.map_structure(lambda x: np.squeeze(x.numpy()), terms)
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
        if deterministic:
            action = self.actor(feature)[0].mode()
            return action, state
        else:
            if self._is_action_discrete:
                act_dist = self.actor(feature)[0]
                action = act_dist.sample(one_hot=False)
                rand_act = tfd.Categorical(tf.zeros_like(act_dist.logits)).sample()
                if getattr(self, '_act_eps', 0):
                    action = tf.where(
                        tf.random.uniform(action.shape[:1], 0, 1) < self._act_eps,
                        rand_act, action)
                logpi = act_dist.log_prob(action)
            else:
                act_dist = self.actor(feature)[0]
                action = act_dist.sample()
                if getattr(self, '_act_eps', 0):
                    action = tf.clip_by_value(
                        tfd.Normal(action, self._act_eps).sample(), -1, 1)
                    mean = act_dist.sample(100)
                    probs1 = act_dist.prob(mean)
                    proxy_dist = tfd.MultivariateNormalDiag(mean, tf.ones_like(mean) * self._act_eps)
                    probs2 = proxy_dist.prob(action)
                    logpi = tf.math.log(tf.reduce_sum(probs1*probs2, axis=0))
                else:
                    logpi = act_dist.log_prob(action)
        
            return action, {'prev_logpi': logpi}, state

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            data = self.dataset.sample()
            log_images = tf.convert_to_tensor(
                self._log_images and i == 0 and self._to_log_images(step), 
                tf.bool)
            terms = self.learn(**data, log_images=log_images)
            terms = {k: v.numpy() for k, v in terms.items()}
            self.store(**terms)
            self._update_target_nets()
        return self.N_UPDATES

    @tf.function
    def _learn(self, obs, prev_action, reward, discount, prev_logpi, log_images, state=None):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(obs)
            if self._burn_in:
                bis = self._burn_in_size
                ss = self._sample_size - self._burn_in_size
                bi_embed, embed = tf.split(embed, [bis, ss], 1)
                bi_prev_action, prev_action = tf.split(prev_action, [bis, ss], 1)
                state, _ = self.rssm.observe(bi_embed, bi_prev_action, state)
                state = tf.nest.pack_sequence_as(state, 
                    tf.nest.map_structure(lambda x: tf.stop_gradient(x[:, -1]), state))
                
                _, obs = tf.split(obs, [bis, ss], 1)
                _, reward = tf.split(reward, [bis, ss], 1)
                _, discount = tf.split(discount, [bis, ss], 1)
                _, prev_logpi = tf.split(prev_logpi, [bis, ss], 1)
            post, prior = self.rssm.observe(embed, prev_action, state)
            feature = self.rssm.get_feat(post)
            obs_pred = self.decoder(feature)
            reward_pred = self.reward(feature)
            likelihoods = AttrDict()
            likelihoods.obs_loss = -tf.reduce_mean(obs_pred.log_prob(obs))
            likelihoods.obs_loss = tf.cast(likelihoods.obs_loss, tf.float32)
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

        with tf.GradientTape() as actor_tape:
            act_dist, terms = self.actor(feature)
            new_action = act_dist.sample()
            new_logpi = act_dist.log_prob(new_action)
            _, temp = self.temperature(feature, new_action)
            q1_with_actor = self.q1(feature, new_action)
            q2_with_actor = self.q2(feature, new_action)
            q_with_actor = tf.minimum(q1_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(temp * new_logpi - q_with_actor)
        
        target_entropy = getattr(self, 'target_entropy', -self._action_dim)
        with tf.GradientTape() as temp_tape:
            if isinstance(self.temperature, (float, tf.Variable)):
                temp = self.temperature
            else:
                log_temp, temp = self.temperature(feature, new_action)
                temp_loss = -tf.reduce_mean(log_temp 
                    * tf.stop_gradient(new_logpi + target_entropy))
                terms['temp'] = temp
                terms['temp_loss'] = temp_loss

        curr_feat = feature[:, :-1]
        next_feat = feature[:, 1:]
        curr_action = prev_action[:, 1:]
        next_action = new_action[:, 1:]
        next_logpi = new_logpi[:, 1:]
        discount = discount[1:] * self._gamma
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        with tf.GradientTape() as q_tape:
            q1 = self.q1(curr_feat, curr_action)
            q2 = self.q2(curr_feat, curr_action)
            q = tf.minimum(q1, q2)
            next_q1 = self.target_q1(next_feat, next_action)
            next_q2 = self.target_q2(next_feat, next_action)
            next_q = tf.minimum(next_q1, next_q2)
            next_value = next_q - temp * next_logpi
            log_ratio = next_logpi - prev_logpi[:, 2:]
            ratio = tf.math.exp(log_ratio)
            returns = retrace_lambda(
                reward[:, :-1], q, next_value, 
                ratio, discount, lambda_=self._lambda, 
                ratio_clip=1, axis=1, tbo=self._tbo)
            # returns = reward[:, :-1] + discount * next_value
            returns = tf.stop_gradient(returns)

            q1_loss = tf.reduce_mean(loss_fn(returns - q1))
            q2_loss = tf.reduce_mean(loss_fn(returns - q2))
            q_loss = q1_loss + q2_loss

        terms['model_norm'] = self._model_opt(model_tape, model_loss)
        terms['actor_norm'], actor_vg = self._actor_opt(actor_tape, actor_loss)
        terms['q_norm'], q_vg = self._q_opt(q_tape, q_loss)
        if not isinstance(self.temperature, (float, tf.Variable)):
            terms['temp_norm'] = self._temp_opt(temp_tape, temp_loss)
        
        terms = dict(
            prior_entropy=prior_dist.entropy(),
            post_entropy=post_dist.entropy(),
            kl=kl,
            q1=q1_with_actor,
            q2=q2_with_actor,
            returns=returns,
            logpi=new_logpi,
            action_entropy=act_dist.entropy(),
            **likelihoods,
            **terms,
            model_loss=model_loss,
            actor_loss=actor_loss,
            q1_loss=q1_loss,
            q2_loss=q2_loss,
            q_loss = q_loss
        )

        if log_images:
            # tf.print(tf.summary.experimental.get_step())
            tf.summary.experimental.set_step(self._env_step)
            tf.summary.histogram(f'{self.name}/returns', returns)
            tf.summary.histogram(f'{self.name}/q', q)
            tf.summary.histogram(f'{self.name}/error', returns-q1)
            self._image_summaries(obs, action, embed, obs_pred)
    
        return terms

    def _image_summaries(self, obs, prev_action, embed, image_pred):
        truth = obs[:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self.rssm.observe(embed[:6, :5], prev_action[:6, :5])
        init = RSSMState(*[v[:, -1] for v in init])
        prior = self.rssm.imagine(prev_action[:6, 5:], init)
        openl = self.decoder(self.rssm.get_feat(prior)).mode()
        # join the first 5 reconstructed images to the imagined subsequent images
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        self.graph_summary('video', 'comp', openl, (1, 6),
            step=self._env_step)

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