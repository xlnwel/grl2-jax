import functools
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.utils import AttrDict, Every
from utility.rl_utils import lambda_return
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
        self._value_opt = DreamerOpt(models=self.value, lr=self._value_lr, return_grads=True)

        self._ckpt_models['model_opt'] = self._model_opt
        self._ckpt_models['actor_opt'] = self._actor_opt
        self._ckpt_models['value_opt'] = self._value_opt

        self._state = None
        self._prev_action = None

        self._obs_shape = env.obs_shape
        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        self._to_log_images = Every(self.LOG_INTERVAL)

        # time dimension must be explicitly specified here
        # otherwise, InaccessibleTensorError arises when expanding rssm
        TensorSpecs = dict(
            obs=((self._sample_size, *self._obs_shape), self._dtype, 'obs'),
            action=((self._sample_size, self._action_dim), self._dtype, 'action'),
            reward=((self._sample_size,), self._dtype, 'reward'),
            discount=((self._sample_size,), self._dtype, 'discount'),
            log_images=(None, tf.bool, 'log_images')
        )
        if self._store_state:
            state_size = self.rssm.state_size
            TensorSpecs['state'] = (RSSMState(
               *[((sz, ), self._dtype, name) 
               for name, sz in zip(RSSMState._fields, state_size)]
            ))

        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    def reset_states(self, state=(None, None)):
        self._state, self._prev_action = state

    def get_states(self):
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
        if deterministic:
            action, self._state = self.action(
                obs, self._state, self._prev_action, deterministic)
        else:
            action, logpi, self._state = self.action(
                obs, self._state, self._prev_action, deterministic)
        
        self._prev_action = tf.one_hot(action, self._action_dim, dtype=self._dtype) \
            if self._is_action_discrete else action
        
        action = np.squeeze(action.numpy()) if has_expanded else action.numpy()
        if deterministic:
            return action
        elif self._store_state:
            return action, {'logpi': logpi.numpy(), 
                **tf.nest.map_structure(lambda x: x.numpy(), self._state._asdict())}
        else:
            return action, {'logpi': logpi.numpy()}
        
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
                action = act_dist.sample(reparameterize=False, one_hot=False)
                rand_act = tfd.Categorical(tf.zeros_like(act_dist.logits)).sample()
                action = tf.where(
                    tf.random.uniform(action.shape[:1], 0, 1) < self._act_eps,
                    rand_act, action)
                logpi = act_dist.log_prob(action)
            else:
                act_dist = self.actor(feature)[0]
                action = act_dist.sample()
                action = tf.clip_by_value(
                    tfd.Normal(action, self._act_eps).sample(), -1, 1)
                logpi = act_dist.log_prob(action)

            return action, logpi, state

    def learn_log(self, step):
        self.global_steps.assign(step)
        for i in range(self.N_UPDATES):
            data = self.dataset.sample()
            if i == 0:
                tf.summary.histogram(f'{self.name}/logpi', data['logpi'], step)
            del data['logpi']
            log_images = tf.convert_to_tensor(
                self._log_images and i == 0 and self._to_log_images(step), 
                tf.bool)
            terms = self.learn(**data, log_images=log_images)
            terms = {k: v.numpy() for k, v in terms.items()}
            self.store(**terms)

    @tf.function
    def _learn(self, obs, action, reward, discount, log_images, state=None):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(obs)
            if self._burn_in:
                bis = self._burn_in_size
                ss = self._sample_size - self._burn_in_size
                burn_in_embed, embed = tf.split(embed, [bis, ss], 1)
                burn_in_action, action = tf.split(action, [bis, ss], 1)
                state, _ = self.rssm.observe(burn_in_embed, burn_in_action, state)
                state = tf.nest.pack_sequence_as(state, 
                    tf.nest.map_structure(lambda x: tf.stop_gradient(x[:, -1]), state))
                
                _, obs = tf.split(obs, [bis, ss], 1)
                _, reward = tf.split(reward, [bis, ss], 1)
                _, discount = tf.split(discount, [bis, ss], 1)
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

        with tf.GradientTape() as actor_tape:
            imagined_feature = self._imagine_ahead(post)
            reward = self.reward(imagined_feature).mode()
            if hasattr(self, 'discount'):
                discount = self.discount(imagined_feature).mean()
            else:
                discount = self._gamma * tf.ones_like(reward)
            value = self.value(imagined_feature).mode()
            # compute lambda return at each imagined step
            returns = lambda_return(
                reward[:-1], value[:-1], discount[:-1], 
                self._lambda, value[-1], axis=0)
            # discount lambda returns based on their sequential order
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(discount[:1]), discount[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            
        with tf.GradientTape() as value_tape:
            value_pred = self.value(imagined_feature)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm, actor_vg = self._actor_opt(actor_tape, actor_loss)
        value_norm, value_vg = self._value_opt(value_tape, value_loss)
        
        act_dist, terms = self.actor(feature)
        terms = dict(
            prior_entropy=prior_dist.entropy(),
            post_entropy=post_dist.entropy(),
            kl=kl,
            value=value,
            returns=returns,
            action_entropy=act_dist.entropy(),
            **likelihoods,
            **terms,
            model_loss=model_loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            model_norm=model_norm,
            actor_norm=actor_norm,
            value_norm=value_norm,
        )

        if log_images:
            tf.summary.experimental.set_step(self.global_steps)
            tf.summary.histogram(f'{self.name}/returns', returns)
            for var, grad in actor_vg:
                tf.summary.histogram(f'grads/{var.name}', grad)
                tf.summary.histogram(f'vars/{var.name}', var)
                tf.summary.scalar(f'grads/{var.name}_mean', tf.reduce_mean(grad))
                tf.summary.scalar(f'grads/{var.name}_std', tf.math.reduce_std(grad))
                tf.summary.scalar(f'grads/{var.name}_sum', tf.math.reduce_sum(grad))
            for var, grad in value_vg:
                tf.summary.histogram(f'grads/{var.name}', grad)
                tf.summary.histogram(f'vars/{var.name}', var)
                tf.summary.scalar(f'grads/{var.name}_mean', tf.reduce_mean(grad))
                tf.summary.scalar(f'grads/{var.name}_std', tf.math.reduce_std(grad))
                tf.summary.scalar(f'grads/{var.name}_sum', tf.math.reduce_sum(grad))
            self._image_summaries(obs, action, embed, obs_pred)
    
        return terms

    def _imagine_ahead(self, post):
        if hasattr(self, 'discount'):   # Omit the last step as it could be done
            post = RSSMState(*[v[:, :-1] for v in post])
        # we merge the time dimension into the batch dimension 
        # since we treat each state as a starting state when imagining
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = RSSMState(*[flatten(x) for x in post])
        policy = lambda state: self.actor(
            tf.stop_gradient(self.rssm.get_feat(state)))[0].sample()
        states = static_scan(
            lambda prev_state, _: self.rssm.img_step(prev_state, policy(prev_state)),
            start, tf.range(self._horizon)
        )
        imagined_features = self.rssm.get_feat(states)
        return imagined_features

    def _image_summaries(self, obs, action, embed, image_pred):
        truth = obs[:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self.rssm.observe(embed[:6, :5], action[:6, :5])
        init = RSSMState(*[v[:, -1] for v in init])
        prior = self.rssm.imagine(action[:6, 5:], init)
        openl = self.decoder(self.rssm.get_feat(prior)).mode()
        # join the first 5 reconstructed images to the imagined subsequent images
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        self.graph_summary(video_summary, ['dreamer/comp', openl, (1, 6)],
            step=self.global_steps)
