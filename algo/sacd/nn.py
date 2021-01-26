import logging
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from algo.sac.nn import Temperature
from algo.dqn.nn import Encoder

logger = logging.getLogger(__name__)


class Actor(Module):
    def __init__(self, config, action_dim, name='actor'):
        super().__init__(name=name)
        
        self._action_dim = action_dim
        prior = np.ones(action_dim, dtype=np.float32)
        prior /= np.sum(prior)
        self.prior = tf.Variable(prior, trainable=False, name='prior')

        self._layers = mlp(
            **config, 
            out_size=action_dim,
            out_dtype='float32',
            name=name)
    
    @property
    def action_dim(self):
        return self._action_dim

    def call(self, x, evaluation=False, epsilon=0, temp=1):
        self.logits = logits = self._layers(x)

        if evaluation:
            if temp == 0:
                dist = tfd.Categorical(logits)
                action = dist.mode()
            else:
                logits = logits / temp
                dist = tfd.Categorical(logits)
                action = dist.sample()
        else:
            if isinstance(epsilon, tf.Tensor) or epsilon:
                scaled_logits = logits / temp
                prior_logits = tf.math.log(tf.maximum(self.prior, 1e-8))
                prior_logits = tf.broadcast_to(prior_logits, logits.shape)
                cond = tf.random.uniform(tf.shape(epsilon), 0, 1) > epsilon
                cond = tf.reshape(cond, (-1, 1))
                logits = tf.where(cond, scaled_logits, prior_logits)
                # scaled_logits = logits / temp
                # cond = tf.random.uniform(tf.shape(epsilon), 0, 1) > epsilon
                # cond = tf.reshape(cond, (-1, 1))
                # logits = tf.where(cond, logits, scaled_logits)

            dist = tfd.Categorical(logits)
            action = dist.sample()

        return action

    def train_step(self, x):
        x = self._layers(x)
        probs = tf.nn.softmax(x)
        logps = tf.math.log(tf.maximum(probs, 1e-8))    # bound logps to avoid numerical instability
        return probs, logps

    def update_prior(self, x, lr):
        self.prior.assign_add(lr * (x - self.prior))


class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)

        self._layers = mlp(
            self._units_list, 
            out_size=action_dim,
            kernel_initializer=self._kernel_initializer,
            activation=self._activation,
            out_dtype='float32')

    def call(self, x, action=None):
        q = self._layers(x)
        if action is not None:
            if action.dtype.is_integer:
                action = tf.one_hot(action, q.shape[-1])
            assert action.shape[1:] == q.shape[1:], (action.shape, q.shape)
            q = tf.reduce_sum(q * action, axis=-1)

        return q


class SAC(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0, return_stats=False, return_eval_stats=False, **kwargs):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        action = self.actor(x, evaluation=evaluation, epsilon=epsilon)
        terms = {}
        if return_eval_stats:
            action, terms = action
            q = self.q(x)
            q = tf.squeeze(q)
            idx = tf.stack([tf.range(action.shape[0]), action], -1)
            q = tf.gather_nd(q, idx)

            action_best_q = tf.argmax(q, 1)
            action_best_q = tf.squeeze(action_best_q)
            terms = {
                'action': action,
                'action_best_q': action_best_q,
                'q': q,
            }
        elif return_stats:
            q = self.q(x, action=action)
            q = tf.squeeze(q)
            terms['q'] = q
            if self.reward_kl:
                kl = -tfd.Categorical(self.actor.logits).entropy()
                if self.temperature.type == 'schedule':
                    _, temp = self.temperature(self._train_step)
                elif self.temperature.type == 'state-action':
                    raise NotImplementedError
                else:
                    _, temp = self.temperature()
                kl = temp * kl
                terms['kl'] = kl
        action = tf.squeeze(action)
        return action, terms

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape
        
        x = self.encoder(x)
        value = self.q(x)
        value = tf.squeeze(value)
        
        return value


def create_components(config, env):
    assert env.is_action_discrete
    config = config.copy()
    action_dim = env.action_dim
    encoder_config = config['encoder']
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
        
    models = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        actor=Actor(actor_config, action_dim),
        q=Q(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_actor=Actor(actor_config, action_dim, name='target_actor'),
        target_q=Q(q_config, action_dim, name='target_q'),
        temperature=Temperature(temperature_config),
    )
    if config['twin_q']:
        models['q2'] = Q(q_config, action_dim, name='q2')
        models['target_q2'] = Q(q_config, action_dim, name='target_q2')

    return models

def create_model(config, env, **kwargs):
    return SAC(config=config, env=env, **kwargs)
