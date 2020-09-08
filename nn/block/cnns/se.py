from nn.utils import *
from nn.block.cnns.utils import *


class SE(layers.Layer):
    def __init__(self, 
                 reduction_ratio,
                 out_activation='sigmoid',
                 name='se', 
                 **kwargs):
        super().__init__(name=name)
        self._reduction_ratio = reduction_ratio
        self._out_activation = out_activation
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
        out_activation = get_activation(self._out_activation)
        self._excitation = [
            layers.Dense(channels // self._reduction_ratio, 
                kernel_initializer=kernel_initializer, activation='relu'),
            layers.Dense(channels, activation=out_activation)
        ]
    
    def call(self, x):
        y = self._squeeze(x)
        for l in self._excitation:
            y = l(y)
        y = tf.reshape(y, (-1, 1, 1, x.shape[-1]))
        return x * y
