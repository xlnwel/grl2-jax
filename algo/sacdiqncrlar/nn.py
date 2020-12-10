import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from algo.sacdiqncrl.nn import Encoder, Actor, Q, Temperature, CRL


class Actor(Module):
    @config
    def __init__(self, action_dim, name='actor'):
        super().__init__(name=name)
        
        self._layers = mlp(self._units_list, 
                            kernel_initializer=self._kernel_initializer,
                            activation=self._activation,
                            name='thunk')
        self._action = mlp(out_size=action_dim, 
                            kernel_initializer=self._kernel_initializer,
                            name='action')
        self._ar = mlp(out_size=self._max_ar, 
                        kernel_initializer=self._kernel_initializer,
                        name='ar')
        self._action_dim = action_dim
    
    @property
    def action_dim(self):
        return self._action_dim
    
    @property
    def max_ar(self):
        return self._max_ar

    def __call__(self, x, evaluation=False, epsilon=0):
        x = self._layers(x)

        def fn(mlp, x):
            logits = mlp(x)
            dist = tfd.Categorical(logits=logits)
            sample = dist.mode() if evaluation else dist.sample()
            if epsilon > 0:
                rand_samp = tfd.Categorical(tf.zeros_like(dist.logits)).sample()
                sample = tf.where(
                    tf.random.uniform(sample.shape, 0, 1) < epsilon,
                    rand_samp, sample)
            return sample
        
        action = fn(self._action, x)

        act_oh = tf.one_hot(action, self._action_dim, dtype=tf.float32)
        x = tf.concat([x, act_oh], axis=-1)
        ar = fn(self._ar, x)

        return action, ar

    def train_step(self, x, action=None):
        x = self._layers(x)
        
        def fn(mlp, x):
            logits = mlp(x)
            probs = tf.nn.softmax(logits)
            logps = tf.math.log(tf.maximum(probs, 1e-8))    # bound logps to avoid numerical instability

            return probs, logps

        act_probs, act_logps = fn(self._action, x)

        if action is None:
            action = tfd.Categorical(probs=act_probs).sample()
        act_oh = tf.one_hot(action, self._action_dim, dtype=tf.float32)
        x = tf.concat([x, act_oh], axis=-1)
        ar_probs, ar_logps = fn(self._ar, x)
        
        return act_probs, act_logps, ar_probs, ar_logps


class SACIQNCRLAR(Ensemble):
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
        action, ar = self.actor(x, evaluation=evaluation, epsilon=epsilon)
        q_action = action * self.actor.max_ar + ar
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
    q_config = config['q']
    temperature_config = config['temperature']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
    q_action_dim = action_dim * actor_config['max_ar']   # q takes into account ar
        
    models = dict(
        encoder=Encoder(config['encoder'], name='encoder'),
        target_encoder=Encoder(config['encoder'], name='target_encoder'),
        actor=Actor(actor_config, action_dim),
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
