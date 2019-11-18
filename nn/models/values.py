import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from nn.layers.func import mlp_layers


def get_valuefunc(type, units_list, output_dim=1, norm=None, name=None, **kwargs):



class V(keras.Model):
    def __init__(self, units_list, output_dim=1, norm=None, name=None, **kwargs):
        super().__init__(name=name)

        self.intra_layers = mlp_layers(units_list, output_dim, norm=norm, **kwargs)

    def call(self, x):
        for l in self.intra_layers:
            x = l(x)
        
        return x


class Q(keras.Model):
    def __init__(self, units_list, output_dim=1, norm=None, name=None, **kwargs):
        super().__init__(name=name)

        self.intra_layers = mlp_layers(units_list, output_dim, norm=norm, **kwargs)

    def call(self, x, a):
        tf.concat([x, a], axis=1)
        for l in self.intra_layers:
            x = l(x)
        
        return x
