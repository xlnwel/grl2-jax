from nn.utils import *
from nn.block.cnns.utils import *


relu = activations.relu

class Residual(layers.Layer):
    def __init__(self, 
                 name='res', 
                 **kwargs):
        super().__init__(name=name)
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)

        kwargs = self._kwargs
        filters = input_shape[-1]
        
        self._conv1 = conv2d(filters, 3, strides=1, padding='same', **kwargs)
        self._conv2 = conv2d(filters, 3, strides=1, padding='same', **kwargs)

    def call(self, x):
        y = relu(x)
        y = self._conv1(y)
        y = relu(y)
        y = self._conv2(y)
        return x + y
