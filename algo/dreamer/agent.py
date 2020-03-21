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
        if hasattr(self, 'terminal'):
            dynamics_models.append(self.terminal)
        opt_name = getattr(self, 'self._optimizer', 'adam')
        DreamerOpt = functools.partial(
            Optimizer, weight_decay=self._weight_decay, clip_norm=self._clip_norm
        )
        self._model_opt = DreamerOpt(self._optimizer, dynamics_models, self._model_lr)
        self._actor_opt = DreamerOpt(self._optimizer, self.actor, self._actor_lr)
        self._value_opt = DreamerOpt(self._optimizer, self.value, self._value_lr)

        self._ckpt_models['model_opt'] = self._model_opt
        self._ckpt_models['actor_opt'] = self._actor_opt
        self._ckpt_models['value_opt'] = self._value_opt

        self.curr_state = None
        self.prev_action = None

        self._obs_shape = (64, 64, 1)#env.obs_shape
        self._action_dim = env.action_space.n

        # time dimension must be specified explicitly here
        # otherwise, InaccessibleTensorError arises when expanding rssm
        # TensorSpecs = dict(
        #     obs=((self._length, *self._obs_shape), tf.float32, 'obs'),
        #     action=((self._length, self._action_dim), tf.float32, 'action'),
        #     reward=((self._length,), tf.float32, 'reward'),
        #     done=((self._length,), tf.float32, 'done'),
        # )

        # self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    def reset_states(self, state, prev_action):
        self.curr_state = state
        self.prev_action = prev_action

    def retrieve_states(self):
        return self.curr_state, self.prev_action

    def __call__(self, obs, done, deterministic=False):
        done = done.astype(np.float32)
        action = self.action(obs, done, deterministic)
        action = np.squeeze(action.numpy())
        return action
        
    @tf.function
    def action(self, obs, done, deterministic=False):
        if self.curr_state is None and self.prev_action is None:
            self.curr_state = self.rssm.get_initial_state(batch_size=len(done))
            self.prev_action = tf.zeros((len(done), self._action_dim))
        obs = tf.expand_dims(obs, 1)
        embed = self.encoder(obs)
        embed = tf.squeeze(embed, 1)
        state, _ = self.rssm.obs_step(self.curr_state, self.prev_action, embed)
        feature = self.rssm.get_feat(state)
        if deterministic:
            action = self.actor(feature).mode()
        else:
            action = self.actor(feature).sample()
            if self._epsilon > 0:
                action = tfd.Normal(action, self._epsilon).sample()
        
        mask = tf.cast(1. - done, self._dtype)[:, None]
        self.curr_state = tf.nest.map_structure(lambda x: x * mask, state)
        self.prev_action = action * mask
        
        return action

    def learn_log(self, step=None):
        for i in range(self._n_updates):
            data = self.dataset.sample()
            terms = self.learn(**data)
            terms = {k: v.numpy() for k, v in terms.items()}
            self.store(**terms)

    # @tf.function
    def _learn(self, obs, action, reward, done):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(obs)
            # print(embed)
            post, prior = self.rssm.observe(embed, action, 
                self.rssm.get_initial_state(batch_size=tf.shape(obs)[0]))
            feature = self.rssm.get_feat(post)
            obs_pred = self.decoder(feature)
            reward_pred = self.reward(feature)
            likelihoods = AttrDict()
            likelihoods.obs_loss = tf.reduce_mean(obs_pred.log_prob(obs))
            # print(likelihoods.obs_loss)
            likelihoods.reward_loss = tf.reduce_mean(reward_pred.log_prob(reward))
            if hasattr(self, 'terminal'):
                term_pred = self.terminal(feature)
                term_target = self._gamma * done
                # print(term_pred.sample())
                # print(term_target)
                # likelihoods.term_loss = (self._term_scale 
                #     * tf.reduce_mean(term_pred.log_prob(term_target)))
                # print(self._term_scale * term_pred.log_prob(term_target))
            prior_dist = self.rssm.get_dist(prior.mean, prior.std)
            post_dist = self.rssm.get_dist(post.mean, post.std)
            kl = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            kl = tf.maximum(kl, self._free_nats)
            model_loss = self._kl_scale * kl - sum(likelihoods.values())

        with tf.GradientTape() as actor_tape:
            imagined_feature = self._imagine_ahead(post)
            reward = self.reward(imagined_feature).mode()
            if hasattr(self, 'terminal'):
                discount = self.terminal(imagined_feature).mean()
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
            kl=kl,
            value=value,
            returns=returns,
            entropy=self.actor(feature).entropy(),
            **likelihoods,
            model_loss=model_loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            model_norm=model_norm,
            actor_norm=actor_norm,
            value_norm=value_norm,
        )

        return terms

    def _imagine_ahead(self, post):
        if hasattr(self, 'terminal'):   # Omit the last step as it could be terminal
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
    steps = 3
    obs_shape = (64, 64, 1)
    act_dim = env.action_space.n
    models = create_model(model_config, obs_shape, act_dim, True)
    tf.random.set_seed(0)
    data = {}
    data['obs'] = tf.random.normal((bs, steps, *obs_shape))
    data['action'] = tf.random.normal((bs, steps, act_dim))
    data['reward'] = tf.random.normal((bs, steps))
    data['done'] = tf.random.normal((bs, steps))
    # print('obs', data['obs'])
    # print('action', data['action'])
    # print('reward', data['reward'])
    # print('done', data['done'])
    agent = Agent(name='dreamer', config=agent_config, models=models, dataset=None, env=env)
    terms = agent._learn(**data)
    print('model_loss', terms['model_loss'])
    print('actor_loss', terms['actor_loss'])
    print('value_loss', terms['value_loss'])
    print('model_norm', terms['model_norm'])
    print('actor_norm', terms['actor_norm'])
    print('value_norm', terms['value_norm'])
    # data['obs'] = tf.random.normal((bs, steps, *obs_shape))
    # data['action'] = tf.random.normal((bs, steps, act_dim))
    # data['reward'] = tf.random.normal((bs, steps))
    # data['done'] = tf.random.normal((bs, steps))
    # terms = agent._learn(**data)
    # print('model_loss', terms['model_loss'])
    # print('actor_loss', terms['actor_loss'])
    # print('value_loss', terms['value_loss'])
    # print('model_norm', terms['model_norm'])
    # print('actor_norm', terms['actor_norm'])
    # print('value_norm', terms['value_norm'])
    