import functools
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
from core.decorator import agent_config, display_model_var_info
from core.optimizers import Optimizer
from algo.dreamer.nn import RSSMState


class Agent(BaseAgent):
    @agent_config
    def __init__(self,
                *,
                dataset,
                env):
        # dataset for input pipline optimization
        self.dataset = dataset

        # optimizer
        dynamics_models = [self.rssm, self.reward]
        if hasattr(self, 'encoder'):
            dynamics_models += [self.encoder, self.decoder]
        opt_name = getattr(self, '_opt_name', 'adam')
        DreamerOpt = functools.partial(
            Optimizer, weight_decay=self._weight_decay, clip_norm=self._clip_norm
        )
        self._model_opt = DreamerOpt(opt_name, dynamics_models, self._model_lr)
        self._actor_opt = DreamerOpt(opt_name, self.actor, self._actor_lr)
        self._value_opt = DreamerOpt(opt_name, self.value, self._value_lr)

        self._ckpt_models['model_opt'] = self._model_opt
        self._ckpt_models['actor_opt'] = self._actor_opt
        self._ckpt_models['value_opt'] = self._value_opt

        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        TensorSpecs = dict(
            state=(env.state_shape, tf.float32, 'state'),
            action=(env.action_shape, tf.float32, 'action'),
            reward=((), tf.float32, 'reward'),
            done=((), tf.float32, 'done'),
        )

        self.learn = build(self._learn, TensorSpecs)

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
            kl = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            kl = tf.maximum(kl, self._free_nats)
            model_loss = self._kl_scale * kl - sum(likelihoods.values())

        with tf.GradientTape() as actor_tape:
            imagined_feature = self._imagine_ahead(post)
            reward = self.reward(imagined_feature).mode()
            if hasattr(self, 'term'):
                discount = self.term(imagined_feature).mean()
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
            value_pred = self.value(imagined_feature[:-1])
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

        print(model_loss)
        print(actor_loss)
        print(value_loss)
        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)
        print(model_norm)
        print(actor_norm)
        print(value_norm)


    def _imagine_ahead(self, post):
        if hasattr(self, 'term'):   # Omit the last step as it could be terminal
            post = RSSMState(*[v[:, :-1] for v in post])
        # we merge the time dimension into the batch dimension 
        # since we treat each state as a starting state when imagining
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = RSSMState(*[flatten(x) for x in post])
        policy = lambda state: self.actor(
            tf.stop_gradient(self.rssm.get_feature(state))).sample()
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
    agent = Agent(name='dreamer', config=agent_config, models=models, dataset=None, env=None)
    tf.random.set_seed(0)
    data = {}
    data['obs'] = tf.random.normal((bs, steps, 64, 64, 3))
    data['action'] = tf.random.normal((bs, steps, act_dim))
    data['reward'] = tf.random.normal((bs, steps))
    agent._learn(**data)