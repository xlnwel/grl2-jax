import functools
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

from utility.display import pwc
from utility.utils import AttrDict
from utility.rl_utils import lambda_return
from utility.tf_utils import static_scan
from utility.schedule import PiecewiseSchedule, TFPiecewiseSchedule
from utility.timer import TBTimer
from utility.graph import video_summary
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, display_model_var_info
from core.optimizer import Optimizer
from algo.dreamer.nn import RSSMState


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        # dataset for input pipline optimization
        self.dataset = dataset
        self._dtype = prec.global_policy().compute_dtype

        # optimizer
        dynamics_models = [self.encoder, self.rssm, self.decoder, self.reward]
        if hasattr(self, 'discount'):
            dynamics_models.append(self.discount)

        DreamerOpt = functools.partial(
            Optimizer, name=self._optimizer, 
            weight_decay=self._weight_decay, clip_norm=self._clip_norm
        )
        self._model_opt = DreamerOpt(models=dynamics_models, lr=self._model_lr)
        self._actor_opt = DreamerOpt(models=self.actor, lr=self._actor_lr)
        self._value_opt = DreamerOpt(models=self.value, lr=self._value_lr)

        self._ckpt_models['model_opt'] = self._model_opt
        self._ckpt_models['actor_opt'] = self._actor_opt
        self._ckpt_models['value_opt'] = self._value_opt

        self.state = None
        self.prev_action = None

        self._obs_shape = env.obs_shape
        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        # time dimension must be specified explicitly here
        # otherwise, InaccessibleTensorError arises when expanding rssm
        TensorSpecs = dict(
            obs=((self._batch_length, *self._obs_shape), self._dtype, 'obs'),
            action=((self._batch_length, self._action_dim), self._dtype, 'action'),
            reward=((self._batch_length,), self._dtype, 'reward'),
            discount=((self._batch_length,), self._dtype, 'discount'),
            log_images=(None, tf.bool, 'log_images')
        )

        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    def reset_states(self, state, prev_action):
        self.state = state
        self.prev_action = prev_action

    def retrieve_states(self):
        return self.state, self.prev_action

    def __call__(self, obs, reset, deterministic=False):
        if self.state is None and self.prev_action is None:
            self.state = self.rssm.get_initial_state(batch_size=tf.shape(reset)[0])
            self.prev_action = tf.zeros((tf.shape(reset)[0], self._action_dim), self._dtype)
        if reset.any():
            mask = tf.cast(1. - reset, self._dtype)[:, None]
            self.state = tf.nest.map_structure(lambda x: x * mask, self.state)
            self.prev_action = self.prev_action * mask
        action, self.state = self.action(
            obs, self.state, self.prev_action, deterministic)
        if self._is_action_discrete:
            self.prev_action = tf.one_hot(action, self._action_dim, dtype=self._dtype)
        else:
            self.prev_action = action
        action = action.numpy()
        return action
        
    @tf.function
    def action(self, obs, state, prev_action, deterministic=False):
        if obs.dtype == np.uint8:
            obs = tf.cast(obs, self._dtype) / 255. - .5
        obs = tf.expand_dims(obs, 1)
        embed = self.encoder(obs)
        embed = tf.squeeze(embed, 1)
        state, _ = self.rssm.obs_step(state, prev_action, embed)
        feature = self.rssm.get_feat(state)
        if deterministic:
            action = self.actor(feature).mode()
        else:
            if self._is_action_discrete:
                act_dist = self.actor(feature)
                action = act_dist.sample(reparameterize=False, one_hot=False)
                rand_act = tfd.Categorical(tf.zeros_like(act_dist.logits)).sample()
                action = tf.where(
                    tf.random.uniform(action.shape[:1], 0, 1) < self._act_epsilon,
                    rand_act, action)
            else:
                action = self.actor(feature).sample()
                action = tf.clip_by_value(tfd.Normal(action, self._act_epsilon).sample(), -1, 1)
            
        return action, state

    def learn_log(self, step):
        self.global_steps.assign(step)
        for i in range(self.N_UPDATES):
            data = self.dataset.sample()
            log_images = tf.convert_to_tensor(
                self._log_images and step % self.LOG_INTERVAL == 0 and i == 0, 
                tf.bool)
            terms = self.learn(**data, log_images=log_images)
            terms = {k: v.numpy() for k, v in terms.items()}
            self.store(**terms)

    @tf.function
    def _learn(self, obs, action, reward, discount, log_images):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(obs)
            post, prior = self.rssm.observe(embed, action)
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
                value[-1], lambda_=self._lambda, axis=0)
            # discount lambda returns based on their sequential order
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(discount[:1]), discount[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            
        with tf.GradientTape() as value_tape:
            value_pred = self.value(imagined_feature)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)
        
        terms = dict(
            prior_entropy=prior_dist.entropy(),
            post_entropy=post_dist.entropy(),
            kl=kl,
            value=value,
            returns=returns,
            action_entropy=self.actor(feature).entropy(),
            **likelihoods,
            model_loss=model_loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            model_norm=model_norm,
            actor_norm=actor_norm,
            value_norm=value_norm,
        )

        if log_images:
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
            tf.stop_gradient(self.rssm.get_feat(state))).sample()
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
        self.graph_summary(video_summary, 'dreamer/comp', openl)


if __name__ == '__main__':
    from utility.yaml_op import load_config
    from algo.dreamer.nn import create_model
    import gym
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    config = load_config('algo/dreamer/config.yaml')
    env_config = config['env']
    model_config = config['model']
    agent_config = config['agent']
    
    env = gym.make('BreakoutNoFrameskip-v4')
    bs = 2
    steps = 5
    obs_shape = (64, 64, 3)
    act_dim = 3
    models = create_model(model_config, obs_shape, act_dim, False)
    np.random.seed(0)
    tf.random.set_seed(0)
    agent = Agent(name='dreamer', config=agent_config, models=models, dataset=None, env=env)
    for i in range(2):
        data = {}
        data['obs'] = np.random.randint(0, 256, (bs, steps, *obs_shape))
        data['obs'] = tf.cast(data['obs'], tf.float32) / 255. - .5
        data['action'] = tf.random.normal((bs, steps, act_dim))
        data['reward'] = tf.random.normal((bs, steps))
        data['discount'] = tf.random.normal((bs, steps))
        done = np.random.randint(0, 2, (bs, ))
        print(done)
        terms = agent._learn(**data)
        tf.random.set_seed(i)
        a = agent(data['obs'][:, i%5], done)
        print("action", a[1])
    # terms = agent._learn(**data)
    # terms = agent._learn(**data)
    # print(agent.actor.variables[-2])
    # np.random.seed(0)
    # tf.random.set_seed(0)
    # obs = np.random.randint(0, 256, (bs, *obs_shape), dtype=np.uint8)
    # reset = np.random.randint(0, 2, size=(bs, ))
    # action = agent(obs, reset, deterministic=False)
    # print(action)
    # action = agent(obs, reset, deterministic=False)
    # print(action)
    # print(state)
    # data['obs'] = tf.random.normal((bs, steps, *obs_shape))
    # data['action'] = tf.random.normal((bs, steps, act_dim))
    # data['reward'] = tf.random.normal((bs, steps))
    # data['reset'] = tf.random.normal((bs, steps))
    # terms = agent._learn(**data)