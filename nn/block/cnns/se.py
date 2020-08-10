from nn.utils import *
from nn.block.cnns.utils import *


relu = activations.relu

class SE(layers.Layer):
    """"""
    def __init__(self, 
                 reduction_ratio,
                 name='se', 
                 **kwargs):
        super().__init__(name=name)
        self._reduction_ratio = reduction_ratio
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)

        kwargs = self._kwargs.copy()    # we cannot modify attribute of the layer in build, which will emit an error when save the model
        time_distributed = kwargs.pop('time_distributed', False)
        kernel_initializer = kwargs.get('kernel_initializer', 'he_uniform')
        GlobalAveragePooling = functools.partial(
            time_dist_fn, layers.GlobalAveragePooling2D, 
            time_distributed=time_distributed)
        channels = input_shape[-1]
        
        self._squeeze = GlobalAveragePooling()
        self._excitation = [
            layers.Dense(channels // self._reduction_ratio, kernel_initializer=kernel_initializer, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ]
    
    def call(self, x):
        y = self._squeeze(x)
        for l in self._excitation:
            y = l(y)
        y = tf.reshape(y, (-1, 1, 1, x.shape[-1]))
        return x * y


class ResidualSE(layers.Layer):
    def __init__(self, 
                 reduction_ratio,
                 name='resse', 
                 **kwargs):
        super().__init__(name=name)
        self._reduction_ratio = reduction_ratio
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)
        
        kwargs = self._kwargs
        filters = input_shape[-1]
        
        self._conv1 = conv2d(filters, 3, strides=1, padding='same', **kwargs)
        self._conv2 = conv2d(filters, 3, strides=1, padding='same', **kwargs)
        self._se = SE(self._reduction_ratio, name='se', **kwargs)

    def call(self, x):
        y = relu(x)
        y = self._conv1(y)
        y = relu(y)
        y = self._conv2(y)
        y = self._se(y)
        return x + y