from nn.utils import *
from nn.block.cnns.utils import *


relu = activations.relu

@register_cnn('ftw')
class FTWCNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 name='ftw', 
                 obs_range=[0, 1], 
                 kernel_initializer='orthogonal',
                 out_size=256,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)
        kwargs['kernel_initializer'] = kernel_initializer
        
        self._conv1 = conv2d(32, 8, strides=4, padding='same', **kwargs)
        self._conv2 = conv2d(64, 4, strides=2, padding='same', **kwargs)
        self._conv3 = conv2d(64, 3, strides=1, padding='same', **kwargs)
        self._conv4 = conv2d(64, 3, strides=1, padding='same', **kwargs)

        self.out_size = out_size
        if self.out_size:
            self._dense = layers.Dense(self.out_size, activation=relu,
                            kernel_initializer=kernel_initializer)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        x = relu(self._conv1(x))
        x = self._conv2(x)
        y = relu(x)
        y = self._conv3(y)
        x = x + y
        y = relu(x)
        y = self._conv4(y)
        x = x + y
        x = relu(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)

        return x