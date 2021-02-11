import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import epsilon_greedy
from utility.tf_distributions import Categorical, TanhBijector, SampleDist
from nn.func import mlp
from algo.sac.nn import Temperature


class Actor(Module):
    @config
    def __init__(self, action_dim, name='actor'):
        super().__init__(name=name)

        """ Network definition """
        out_size = 2*action_dim
        self._layers = mlp(self._units_list, 
                            out_size=out_size,
                            activation=self._activation)
        
    @tf.function
    def __call__(self, x, evaluation=False, epsilon=0, temp=1):
        if evaluation:
            action = self.step(x, temp=temp)[0].mode()
        else:
            act_dist = self.step(x, temp=temp)[0]
            action = act_dist.sample()
            if isinstance(epsilon, tf.Tensor) or epsilon:
                action = epsilon_greedy(action, epsilon, False)
        
        return action

    @tf.Module.with_name_scope
    def step(self, x, temp=1):
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
            dist = tfd.Normal(mean, std*temp)
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
    
    def call(self, x):
        x = self._layers(x)
        rbd = 0 if x.shape[-1] == 1 else 1  # #reinterpreted batch dimensions
        x = tf.squeeze(x)
        return tfd.Independent(tfd.Normal(x, 1), rbd)




class SAC(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0, **kwargs):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 2, x.shape
        
        action = self.actor(x, evaluation=evaluation, epsilon=epsilon)
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
        
    return dict(
        actor=Actor(actor_config, action_dim, is_action_discrete),
        q=Q(q_config, 'q'),
        q2=Q(q_config, 'q2'),
        value=Q(q_config, 'v'),
        target_value=Q(q_config, 'target_v'),
        temperature=Temperature(temperature_config),
    )

def create_model(config, env, **kwargs):
    return SAC(config, env, **kwargs)
