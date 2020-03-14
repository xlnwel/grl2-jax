import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.utils import AttrDict
from utility.rl_utils import lambda_return
from utility.tf_utils import static_scan
from utility.schedule import PiecewiseSchedule, TFPiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from core.optimizers import Optimizer
from algo.dreamer.nn import RSSMState


class Agent(BaseAgent):
    @agent_config
    def __init__(self):
        #         *,
        #         dataset,
        #         env):
        # # dataset for input pipline optimization
        # self.dataset = dataset

        # optimizer
        self._model_opt = Optimizer('adam', self.rssm, self._model_lr)
        self._value_opt = Optimizer('adam', self.value, self._value_lr)
        self._actor_opt = Optimizer('adam', self.actor, self._actor_lr)

        # self._action_dim = env.action_dim
        # self._is_action_discrete = env.is_action_discrete

        TensorSpecs = dict(

        )

        # self.learn = build(self._learn, TensorSpecs)

    @tf.function
    def _learn(self, **data):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(data['obs']) if hasattr(self, 'encoder') else data['obs']
            post, prior = self.rssm.observe(embed, data['action'])
            feature = self.rssm.get_feature(post)
            obs_pred = self.decoder(feature) if hasattr(self, 'decoder') else feature
            reward_pred = self.reward(feature)
            likelihoods = AttrDict()
            if hasattr(self, 'decoder'):
                likelihoods.obs = tf.reduce_mean(obs_pred.log_prob(data['obs']))
            likelihoods.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
            if hasattr(self, 'term'):
                term_pred = self.term(feature)
                term_target = data['done']
                likelihoods.term = self.term_scale * tf.reduce_mean(term_pred.log_prob(term_target))
            prior_dist = self.rssm.get_dist(prior.mean, prior.std)
            post_dist = self.rssm.get_dist(post.mean, post.std)
            kl = tf.reduce_mean(tfd.kl_divergence, post_dist, prior_dist)
            kl = tf.maximum(kl, self.free_nats)
            model_loss = self.kl_scale * kl - sum(likelihoods.values())

        with tf.GradientTape() as actor_tape:
            imagined_feature = self._imagine_ahead(post)
            reward = self.reward(imagined_feature).mode()
            if hasattr(self, 'term'):
                discount = self.term(imagined_feature).mean()
            else:
                discount = self._gamma * tf.ones_like(reward)
            value = self.value(imagined_feature).mode()
            returns = lambda_return(
                reward[:-1], value[:-1], discount[:-1], 
                value[-1], lambda_=self._lambda, axis=0)



    def _imagine_ahead(self, post):
        if hasattr(self, 'term'):   # Omit the last step as it could be terminal
            post = RSSMState(*[v[:, :-1] for v in post])
        # we merge the time dimension into the batch dimension 
        # as we treat each state as a starting state when imagining
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = RSSMState(*[flatten(x) for x in post])
        policy = lambda state: self.actor(tf.stop_gradient(self.rssm.get_feature(state))).sample()
        states = static_scan(
            lambda prev_state, _: self.rssm.img_step(prev_state, policy(prev_state)),
            start, tf.range(self._horizon)
        )
        imagined_features = self.rssm.get_feature(states)
        return imagined_features


if __name__ == '__main__':
    from utility.yaml_op import load_config
    from algo.dreamer.nn import create_model
    config = load_config('algo/dreamer/config.yaml')
    env_config = config['env']
    model_config = config['model']
    agent_config = config['agent']
    replay_config = config.get('buffer') or config.get('replay')
    
    replay_config['batch_size'] = bs = 2
    steps = 3
    state_shape = (3,)
    act_dim = 2
    embed_dim = 3
    agent_config['horizon'] = 4
    model_config['rssm'] = dict(
        stoch_size=3, deter_size=2, hidden_size=2, activation='elu'
    )
    models = create_model(model_config, state_shape, act_dim, True)
    tf.random.set_seed(0)
    agent = Agent(name='dreamer', config=agent_config, models=models)
    tf.random.set_seed(0)
    rssm = agent.rssm
    action = tf.random.normal((bs, steps, act_dim))
    embed = tf.random.normal((bs, steps, embed_dim))
    
    post, prior = rssm.observe(embed, action)
    
    feat = agent._imagine_ahead(post)
    r = agent.reward(feat).mode()
    print(r)
    v = agent.value(feat).mode()
    print(v)
    discount = tf.ones_like(r) * .99
    returns = lambda_return(r[:, :-1], v[:, :-1], discount[:, :-1], v[:, -1], .95, 1)
    print(returns)