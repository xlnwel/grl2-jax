import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.module import Module
from core.decorator import config
from nn.func import mlp


class Actor(Module):
    @config
    def __init__(self, action_dim, name='actor'):
        super().__init__(name=name)
        
        self._layers = mlp(self._units_list, 
                            out_size=action_dim,
                            norm=self._norm, 
                            activation=self._activation)
    
    def __call__(self, x, deterministic=False, epsilon=0):
        if len(x.shape) % 2 == 1:
            x = np.expand_dims(x, 0)

        action = self.action(x, deterministic, epsilon)
        action = np.squeeze(action.numpy())

        return action

    @tf.function(experimental_relax_shapes=True)
    def action(self, x, deterministic=False, epsilon=0):
        x = self._layers(x)

        dist = tfd.Categorical(logits=x)
        action = dist.mode() if deterministic else dist.sample()
        if epsilon > 0:
            rand_act = tfd.Categorical(tf.zeros_like(dist.logits)).sample()
            action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)

        return action

    def train_step(self, x):
        x = self._layers(x)
        probs = tf.nn.softmax(x)
        # logps = tf.math.log_softmax(x)
        logps = tf.math.log(probs + tf.cast(probs == 0., tf.float32) * 1e-8)
        return probs, logps

class SoftQ(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)

        self._layers = mlp(self._units_list, 
                            out_size=action_dim,
                            norm=self._norm, 
                            activation=self._activation)

    def __call__(self, x, a=None):
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
            self._layer = layers.Dense(1)
        elif self._temp_type == 'variable':
            self._log_temp = tf.Variable(np.log(self._value), dtype=tf.float32)
        else:
            raise NotImplementedError(f'Error temp type: {self._temp_type}')
    
    def __call__(self, x=None, a=None):
        if self._temp_type == 'state-action':
            x = tf.concat([x, a], axis=-1)
            x = self._layer(x)
            log_temp = -tf.nn.softplus(x)
            log_temp = tf.squeeze(log_temp)
        else:
            log_temp = self._log_temp
        temp = tf.exp(log_temp)
    
        return log_temp, temp


def create_model(config, env):
    assert env.is_action_discrete
    action_dim = env.action_dim
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    actor = Actor(actor_config, action_dim)
    q1 = SoftQ(q_config, action_dim, 'q1')
    q2 = SoftQ(q_config, action_dim, 'q2')
    target_q1 = SoftQ(q_config, action_dim, 'target_q1')
    target_q2 = SoftQ(q_config, action_dim, 'target_q2')
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    return dict(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        temperature=temperature,
    )
