import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import logpi_correction
from utility.tf_distributions import Categorical, TanhBijector, SampleDist
from nn.func import mlp


class Actor(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        """ Network definition """
        out_size = action_dim if is_action_discrete else 2*action_dim
        self._layers = mlp(self._units_list, 
                            out_size=out_size,
                            activation=self._activation)

        self._is_action_discrete = is_action_discrete
        
    @tf.function
    def __call__(self, x, deterministic=False, epsilon=0):
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


class Q(Module):
    @config
    def __init__(self, name='q'):
        super().__init__(name=name)

        self._layers = mlp(self._units_list,
                            out_size=1,
                            activation=self._activation)
    
    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self._layers(x)
        rbd = 0 if x.shape[-1] == 1 else 1  # #reinterpreted batch dimensions
        x = tf.squeeze(x)
        return tfd.Independent(tfd.Normal(x, 1), rbd)


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
        assert x.shape.ndims == 2, x.shape
        
        action = self.actor(x, deterministic=deterministic, epsilon=epsilon)
        action = tf.squeeze(action)

        return action

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 2, x.shape
        
        value = self.q(x).mode()
        value = tf.squeeze(value)

        return value


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    return dict(
        actor=Actor(actor_config, action_dim, is_action_discrete),
        q=Q(q_config, 'q'),
        q2=Q(q_config, 'q2'),
        value=Q(q_config, 'v'),
        target_value=Q(q_config, 'target_v'),
        temperature=temperature,
    )

def create_model(config, env, **kwargs):
    return SAC(config, env, **kwargs)
