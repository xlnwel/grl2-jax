import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from nn.layers.func import mlp_layers
from utility.tf_distributions import DiagGaussian, Categorical


def get_policy(is_action_discrete, units_list, out_dim, simple_logstd=True, norm=None, name=None, **kwargs):
    if is_action_discrete:
        return DiscretePolicy(units_list, out_dim, norm=norm, name=name, **kwargs)
    else:
        return ContinuousPolicy(units_list, out_dim, simple_logstd, norm=norm, name=name, **kwargs)


class ContinuousPolicy(keras.Model):
    def __init__(self, units_list, out_dim, simple_logstd, norm=None, name=None, **kwargs):
        super().__init__(name=name)

        self.simple_logstd = simple_logstd

        self.intra_layers = mlp_layers(units_list, norm=norm, **kwargs)
        self.mu = layers.Dense(out_dim)
        if self.simple_logstd:
            self.logstd = tf.Variable(initial_value=np.zeros(out_dim), 
                                    dtype=tf.float32, 
                                    trainable=True, 
                                    name=f'{name}/logstd')
        else:
            self.logstd = layers.Dense(out_dim)
        
        self.ActionDistributionType = DiagGaussian

    def call(self, x):
        for l in self.intra_layers:
            x = l(x)
        if self.simple_logstd:
            return self.mu(x)
        else:
            return self.mu(x), self.logstd(x)


class DiscretePolicy(keras.Model):
    def __init__(self, units_list, out_dim, norm=None, name=None, **kwargs):
        super().__init__(name=name)

        self.intra_layers = mlp_layers(units_list, out_dim=out_dim, norm=norm, **kwargs)

        self.ActionDistributionType = Categorical

    def call(self, x):
        for l in self.intra_layers:
            x = l(x)

        return x

