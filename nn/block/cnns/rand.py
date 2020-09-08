from nn.utils import *
from nn.block.cnns.utils import *
from nn.block.cnns.res import ResidualV1, ResidualV2


@register_cnn('rand')
class RandCNN(layers.Layer):
    def __init__(self,
                 *,
                 time_distributed=False,
                 obs_range=[0, 1],
                 kernel_size=3,
                 strides=1,
                 kernel_initializer='glorot_normal',
                 name='rand'):
        super().__init__(name=name)
        self._obs_range = obs_range
        self._kernel_size = kernel_size
        self._kernel_initializer = kernel_initializer
        self._time_distributed = time_distributed

    def build(self, input_shape):
        filters = input_shape[-1]
        self._layer = conv2d(
            filters, 
            self._kernel_size, 
            padding='same', 
            kernel_initializer=self._kernel_initializer,
            trainable=False,
            use_bias=False,
            time_distributed=self._time_distributed)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        x = self._layer(x)
        return x