import tensorflow as tf

from core.module import Module
from nn.mlp import MLP
from nn.registry import nn_registry


@nn_registry.register('hyper')
class HyperNet(Module):
    def __init__(self, name='hyper_net', **config):
        super().__init__(name=name)

        config = config.copy()
        self.w_in = config.pop('w_in')
        self.w_out = config.pop('w_out')
        if self.w_in is None:
            config['out_size'] = self.w_out
        else:
            config['out_size'] = self.w_in * self.w_out

        self._layers = MLP(
            **config, 
            name=self.name
        )

    def call(self, x):
        x = self._layers(x)
        if self.w_in is None:
            x = tf.reshape(x, (-1, *x.shape[1:-1], self.w_out))
        else:
            x = tf.reshape(x, (-1, *x.shape[1:-1], self.w_in, self.w_out))

        return x
