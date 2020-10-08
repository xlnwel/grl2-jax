import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from algo.sacdiqn.nn import Encoder, Actor, Q, Temperature, SACIQN


class Transition(Module):
    def __init__(self, config, name='trans'):
        super().__init__(name=name)
        self._config = config
        
    def build(self, input_shape):
        config = self._config.copy()
        if 'dense' in config['layer_type']:
            config['out_size'] = input_shape[-1]
            print(input_shape)
        self._layers = mlp(
            **config,
            out_dtype='float32',
            name=self.name,
        )

    def call(self, x, a, training=False):
        if 'conv2d' in self._config['layer_type']:
            a = a[..., None, None, :]
            m = np.ones_like(x.shape)
            m[-3:-1] = x.shape[-3:-1]
            a = tf.tile(a, m)
        assert a.shape[1:-1] == x.shape[1:-1], (x.shape, a.shape)
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x, training=training)

        return x

class Reward(Module):
    def __init__(self, config, name='reward'):
        super().__init__(name=name)
        self._layers = mlp(
            **config,
            out_size=1,
            out_dtype='float32',
            name=name
        )
    
    def call(self, x, a, training=False):
        x = tf.concat([x, a], axis=-1)
        r = self._layers(x, training=training)

        return r


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    encoder_config = config['encoder']
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    trans_config = config['transition']
    reward_config = config['reward']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    models = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        actor=Actor(actor_config, action_dim, name='actor'),
        target_actor=Actor(actor_config, action_dim, name='target_actor'),
        q=Q(q_config, action_dim, name='q'),
        target_q=Q(q_config, action_dim, name='target_q'),
        temperature=temperature,
        trans=Transition(trans_config, name='trans'),
        reward=Reward(reward_config, name='reward'),
    )
    if config['twin_q']:
        models['q2'] = Q(q_config, action_dim, name='q2')
        models['target_q2'] = Q(q_config, action_dim, name='target_q2')

    return models

def create_model(config, env, **kwargs):
    return SACIQN(config=config, env=env, model_fn=create_components, **kwargs, name='saciqnmdp')
