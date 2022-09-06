import tensorflow as tf

from nn.registry import nn_registry
from nn.utils import get_activation, get_initializer


@nn_registry.register('var')
class Variable(tf.Module):
    def __init__(
        self, 
        initializer, 
        scale=1, 
        shape=(), 
        activation=None, 
        name=None
    ):
        super().__init__(name)

        self._var = tf.Variable(scale*get_initializer(initializer)(shape), name=name)
        self.activation = get_activation(activation)
    
    def __call__(self):
        x = self.activation(self._var)
        return x
