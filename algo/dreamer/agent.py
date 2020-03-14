import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.utils import AttrDict
from utility.rl_utils import n_step_target, transformed_n_step_target
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
    def __init__(self,
                *,
                name,
                config,
                models,
                dataset,
                env):
        # dataset for input pipline optimization
        self.dataset = dataset

        # optimizer
        self._model_opt = Optimizer(learning_rate=self.actor_lr)
        self._value_opt = Optimizer(learning_rate=self.value_lr)
        self._actor_opt = Optimizer(learning_rate=self.actor_lr)

        self._action_dim = env.action_dim
        self._is_action_discrete = env.is_action_discrete

        TensorSpecs = dict(

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
            div = tf.reduce_mean(tfd.kl_divergence, post_dist, prior_dist)
            div = tf.maximum(div, self.free_nats)
            model_loss = self.kl_scale * div - sum(likelihoods.values())
            model_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as actor_tape:
            obs_feature = self._imagine_ahead(post)
            reward = self.reward(obs_feature).mode()

    def _imagine_ahead(self, post):
        if hasattr(self, 'term'):   # Omit the last step as it could be terminal
            post = RSSMState(*[v[:, :-1] for v in post])
        policy = lambda state: self.actor(tf.stop_gradient(self.rssm.get_feature(state))).sample()
        states = static_scan(
            lambda prev_state, _: self.rssm.img_step(prev_state, policy(prev_state))
        )