import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module
from core.decorator import config
from utility.rl_utils import logpi_correction
from utility.tf_distributions import Categorical, TanhBijector, SampleDist
from nn.func import mlp


class Actor(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        """ Network definition """
        out_dim = action_dim if is_action_discrete else 2*action_dim
        self._layers = mlp(self._units_list, 
                            out_dim=out_dim,
                            activation=self._activation)

        self._is_action_discrete = is_action_discrete


    def __call__(self, x, deterministic=False, epsilon=0):
        if len(x.shape) % 2 == 1:
            x = np.expand_dims(x, 0)

        action = self.action(x, deterministic=deterministic, epsilon=epsilon)
        action = np.squeeze(action.numpy())

        return action
        
    @tf.function
    def action(self, x, deterministic=False, epsilon=0):
        if deterministic:
            action = self.step(x)[0].mode()
        else:
            act_dist = self.step(x)[0]
            action = act_dist.sample()
            if epsilon:
                action = tf.clip_by_value(
                    tfd.Normal(action, self._act_eps).sample(), -1, 1)
        
        return action

    @tf.Module.with_name_scope
    def step(self, x):
        x = self._layers(x)

        if self._is_action_discrete:
            dist = Categorical(x)
            terms = {}
        else:
            raw_init_std = np.log(np.exp(self._init_std) - 1)
            mean, std = tf.split(x, 2, -1)
            # https://www.desmos.com/calculator/gs6ypbirgq
            # we bound the mean to [-5, +5] to avoid numerical instabilities 
            # as atanh becomes difficult in highly saturated regions
            # interestingly, algo/sac does not suffer this problem
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = SampleDist(dist)
            terms = dict(raw_act_std=std)

        return dist, terms

class Value(Module):
    @config
    def __init__(self, name='q'):
        super().__init__(name=name)

        self._layers = mlp(self._units_list,
                            out_dim=1,
                            activation=self._activation)
    
    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self._layers(x)
        rbd = 0 if x.shape[-1] == 1 else 1  # #reinterpreted batch dimensions
        x = tf.squeeze(x)
        return tfd.Independent(tfd.Normal(x, 1), rbd)

        return x


class Temperature(Module):
    def __init__(self, config, name='temperature'):
        super().__init__(name=name)

        self.temp_type = config['temp_type']

        if self.temp_type == 'state-action':
            self.intra_layer = layers.Dense(1)
        elif self.temp_type == 'variable':
            self.log_temp = tf.Variable(0., dtype=global_policy().compute_dtype)
        else:
            raise NotImplementedError(f'Error temp type: {self.temp_type}')
    
    def __call__(self, x, a):
        if self.temp_type == 'state-action':
            x = tf.concat([x, a], axis=-1)
            x = self.intra_layer(x)
            log_temp = -tf.nn.softplus(x)
            log_temp = tf.squeeze(log_temp)
        else:
            log_temp = self.log_temp
        temp = tf.exp(log_temp)
    
        return log_temp, temp


def create_model(config, action_dim, is_action_discrete):
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    return dict(
        actor=Actor(actor_config, action_dim, is_action_discrete),
        q1=Value(q_config, 'q1'),
        q2=Value(q_config, 'q2'),
        value=Value(q_config, 'v'),
        target_value=Value(q_config, 'target_v'),
        temperature=temperature,
    )
