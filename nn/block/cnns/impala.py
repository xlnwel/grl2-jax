from nn.utils import *
from nn.block.cnns.utils import *
from nn.block.cnns.res import Residual
from nn.block.cnns.se import ResidualSE

relu = activations.relu


@register_cnn('impala')
class IMPALACNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 channels=[16, 32, 32],
                 name='impala', 
                 kernel_initializer='orthogonal',
                 out_size=256,
                 block=Residual,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)

        self._conv_layers = []
        for i, c in enumerate(channels):
            self._conv_layers += [
                conv2d(c, 3, strides=1, padding='same', **kwargs),
                maxpooling2d(3, strides=2, padding='same', time_distributed=time_distributed),
                block(name=f'block{i}_{c}_1', **kwargs),
                block(name=f'block{i}_{c}_2', **kwargs),
            ]

        self.out_size = out_size
        if self.out_size:
            self._dense = layers.Dense(self.out_size, activation=relu)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = relu(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)

        return x


""" Impala with Squeeze&Excitation """
@register_cnn('impalase')
class ImpalaSECNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 channels=[16, 32, 32],
                 reduction_ratio=1,
                 name='impalase', 
                 kernel_initializer='orthogonal',
                 out_size=256,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)

        self._conv_layers = []
        for i, c in enumerate(channels):
            self._conv_layers += [
                conv2d(c, 3, strides=1, padding='same', **kwargs),
                maxpooling2d(3, strides=2, padding='same', time_distributed=time_distributed),
                ResidualSE(reduction_ratio, name=f'resse{i}_{c}_1', **kwargs),
                ResidualSE(reduction_ratio, name=f'resse{i}_{c}_2', **kwargs),
            ]

        self.out_size = out_size
        if self.out_size:
            self._dense = layers.Dense(self.out_size, activation=relu)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = relu(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)

        return x