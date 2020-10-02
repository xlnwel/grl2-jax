import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp, cnn


class Encoder(Module):
    def __init__(self, config, name='encoder'):
        super().__init__(name=name)
        self._layers = cnn(**config)

    def call(self, x, **kwargs):
        x = self._layers(x, **kwargs)
        return x


class Actor(Module):
    def __init__(self, config, action_dim, name='actor'):
        super().__init__(name=name)
        
        self._layers = mlp(
            **config, 
            out_size=action_dim,
            name='name')
        self._action_dim = action_dim
    
    @property
    def action_dim(self):
        return self._action_dim

    def call(self, x, deterministic=False, epsilon=0, return_stats=False):
        x = self._layers(x)

        dist = tfd.Categorical(logits=x)
        action = dist.mode() if deterministic else dist.sample()
        if epsilon > 0:
            rand_act = tfd.Categorical(tf.zeros_like(dist.logits)).sample()
            action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)

        if return_stats:
            prob = dist.prob(action)
            return action, {'prob': prob}
        else:
            return action

    def train_step(self, x):
        x = self._layers(x)
        probs = tf.nn.softmax(x)
        logps = tf.math.log(tf.maximum(probs, 1e-8))    # bound logps to avoid numerical instability
        return probs, logps


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


class Temperature(Module):
    @config
    def __init__(self, name='temperature'):
        super().__init__(name=name)

        if self._temp_type == 'state-action':
            from nn.utils import get_initializer
            kernel_initializer = get_initializer('orthogonal', gain=.01)
            self._layer = layers.Dense(1, kernel_initializer=kernel_initializer)
        elif self._temp_type == 'variable':
            self._log_temp = tf.Variable(
                np.log(self._value), dtype=tf.float32, name='log_temp')
        else:
            raise NotImplementedError(f'Error temp type: {self._temp_type}')
    
    def call(self, x=None, a=None):
        if self._temp_type == 'state-action':
            x = tf.concat([x, a], axis=-1)
            x = self._layer(x)
            log_temp = -tf.nn.softplus(x)
            log_temp = tf.squeeze(log_temp)
        else:
            log_temp = self._log_temp
        temp = tf.exp(log_temp)
    
        return log_temp, temp


class SAC(Ensemble):
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
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    models = dict(
        encoder=Encoder(config['encoder'], name='encoder'),
        target_encoder=Encoder(config['encoder'], name='target_encoder'),
        actor=Actor(actor_config, action_dim),
        q=Q(q_config, action_dim, name='q'),
        target_q=Q(q_config, action_dim, name='target_q'),
        temperature=temperature,
    )
    if config['twin_q']:
        models['q2'] = Q(q_config, action_dim, name='q2')
        models['target_q2'] = Q(q_config, action_dim, name='target_q2')

    return models

def create_model(config, env, **kwargs):
    return SAC(config, env, **kwargs)
