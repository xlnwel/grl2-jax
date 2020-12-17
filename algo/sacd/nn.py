import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp, cnn
from algo.sac.nn import Temperature
from algo.dqn.nn import Encoder

logger = logging.getLogger(__name__)


class Actor(Module):
    def __init__(self, config, action_dim, name='actor'):
        super().__init__(name=name)
        
        prior = np.ones(action_dim, dtype=np.float32)
        prior /= np.sum(prior)
        self.prior = tf.Variable(prior, trainable=False, name='prior')
        act_temp = config.pop('act_temp', 1.)
        if isinstance(act_temp, (list, tuple, np.ndarray)):
            act_temp = np.expand_dims(act_temp, axis=-1)
        self.act_inv_temp = 1. / act_temp
        self.eval_act_temp = config.pop('eval_act_temp', .5)
        logger.info(f'{self.name} action temperature: {np.squeeze(act_temp)}\n'
            f'action temperature at evaluation: {self.eval_act_temp}')
        self._layers = mlp(
            **config, 
            out_size=action_dim,
            name=name)
        self._action_dim = action_dim
    
    @property
    def action_dim(self):
        return self._action_dim

    def call(self, x, evaluation=False, epsilon=0, return_distribution=False):
        self.logits = logits = self._layers(x)

        if evaluation:
            if self.eval_act_temp == 0:
                dist = tfd.Categorical(logits)
                action = dist.mode()
            else:
                logits = logits / self.eval_act_temp
                dist = tfd.Categorical(logits)
                action = dist.sample()
        else:
            if isinstance(epsilon, tf.Tensor) or epsilon:
                # scaled_logits = logits * self.act_inv_temp
                # prior_logits = tf.math.log(tf.maximum(self.prior, 1e-8))
                # prior_logits = tf.broadcast_to(prior_logits, logits.shape)
                # cond = tf.random.uniform(tf.shape(epsilon), 0, 1) > epsilon
                # cond = tf.reshape(cond, (-1, 1))
                # logits = tf.where(cond, scaled_logits, prior_logits)
                scaled_logits = logits * self.act_inv_temp
                cond = tf.random.uniform(tf.shape(epsilon), 0, 1) > epsilon
                cond = tf.reshape(cond, (-1, 1))
                logits = tf.where(cond, logits, scaled_logits)

            dist = tfd.Categorical(logits)
            action = dist.sample()

        if return_distribution:
            return action, dist
        else:
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

    def call(self, x, a=None):
        q = self._layers(x)
        if a is not None:
            if len(a.shape) < len(q.shape):
                a = tf.one_hot(a, q.shape[-1])
            assert a.shape[1:] == q.shape[1:]
            q = tf.reduce_sum(q * a, axis=-1)

        return q


class SAC(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        action = self.actor(x, evaluation=evaluation, epsilon=epsilon)
        action = tf.squeeze(action)

        return action, {}

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
    action_dim = env.action_dim
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
        
    models = dict(
        encoder=Encoder(config['encoder'], name='encoder'),
        target_encoder=Encoder(config['encoder'], name='target_encoder'),
        actor=Actor(actor_config, action_dim),
        q=Q(q_config, action_dim, name='q'),
        target_q=Q(q_config, action_dim, name='target_q'),
        temperature=Temperature(temperature_config),
    )
    if config['twin_q']:
        models['q2'] = Q(q_config, action_dim, name='q2')
        models['target_q2'] = Q(q_config, action_dim, name='target_q2')

    return models

def create_model(config, env, **kwargs):
    return SAC(config, env, **kwargs)
