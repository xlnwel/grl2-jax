import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from algo.sacdiqncrl.nn import Encoder, Actor, Q, Temperature, CRL


class ActionRepetion(Module):
    def __init__(self, config, name='ar'):
        super().__init__(name=name)
        self._layers = mlp(
            **config,
            out_dtype='float32',
            name='ar'
        )
        self._max_ar = config['out_size']
    
    @property
    def max_ar(self):
        return self._max_ar

    def __call__(self, x, action):
        assert action.shape.ndims == x.shape.ndims, (action.shape, x.shape)
        x = tf.concat([x, action], axis=-1)
        logits = self._layers(x)
        ar_dist = tfd.Categorical(logits)
        return ar_dist

    def train_step(self, x, action, ar=None):
        x = tf.concat([x, action], axis=-1)
        logits = self._layers(x)
        probs = tf.nn.softmax(logits)
        logps = tf.math.log(tf.maximum(probs, 1e-8))    # bound logps to avoid numerical instability
        return probs, logps


class SACIQNCRLAR(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        action = self.actor(x, deterministic=deterministic, epsilon=epsilon)
        action_oh = tf.one_hot(action, self.actor.action_dim)
        ar_dist = self.ar(x, action_oh)
        ar = ar_dist.mode() if deterministic else ar_dist.sample()
        q_action = action * self.ar.max_ar + ar
        _, qtv = self.q(x, action=q_action)
        action = tf.squeeze(action)
        ar = tf.squeeze(ar)
        qtv = tf.squeeze(qtv)
        assert action.shape.ndims == ar.shape.ndims == qtv.shape.ndims - 1 == 0, \
            (action.shape, ar.shape, qtv.shape)

        return action, ar, {'qtv': qtv, 'ar': ar}


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    actor_config = config['actor']
    ar_config = config['ar']
    q_config = config['q']
    temperature_config = config['temperature']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
    q_action_dim = action_dim * ar_config['out_size']   # q takes into account ar
        
    models = dict(
        encoder=Encoder(config['encoder'], name='encoder'),
        target_encoder=Encoder(config['encoder'], name='target_encoder'),
        actor=Actor(actor_config, action_dim),
        ar=ActionRepetion(ar_config),
        crl=CRL(config['crl']),
        q=Q(q_config, q_action_dim, name='q'),
        target_q=Q(q_config, q_action_dim, name='target_q'),
        temperature=temperature,
    )
    if config['twin_q']:
        models['q2'] = Q(q_config, q_action_dim, name='q2')
        models['target_q2'] = Q(q_config, q_action_dim, name='target_q2')

    return models

def create_model(config, env, **kwargs):
    return SACIQNCRLAR(config, env, **kwargs)
