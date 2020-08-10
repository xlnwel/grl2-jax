from nn.utils import *
from nn.layers import Layer
from nn.block.cnns.utils import *
from nn.block.cnns.se import *


""" MobileNet """
class InvertedResidualSE(layers.Layer):
    def __init__(self, 
                 strides,
                 out_size,
                 expansion_ratio,
                 reduction_ratio=None,
                 name='inv_resse', 
                 **kwargs):
        super().__init__(name=name)
        self._strides = strides
        self._out_size = out_size
        self._expansion_ratio = expansion_ratio
        self._reduction_ratio = reduction_ratio
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)
        
        kwargs = self._kwargs.copy()
        conv_kwargs = kwargs.copy()
        conv_kwargs['layer_type'] = conv2d
        conv_kwargs['strides'] = 1
        conv_kwargs['padding'] = 'same'
        conv_kwargs['use_bias'] = False
        conv_kwargs['norm_kwargs'] = dict(
            epsilon=1e-3,
            momentum=.999
        )
        dc_kwargs = conv_kwargs.copy()
        dc_kwargs['layer_type'] = depthwise_conv2d
        dc_kwargs['strides'] = self._strides
        dc_kwargs['depthwise_initializer'] = dc_kwargs.pop('kernel_initializer')
        in_size = input_shape[-1]
        hidden_size = in_size * self._expansion_ratio
        
        self._expand = Layer(hidden_size, 1, name='expand', **conv_kwargs)
        self._depthwise = Layer(3, name='depthwise', **dc_kwargs)
        if self._reduction_ratio:
            self._se = SE(self._reduction_ratio, name='se', **kwargs)
        conv_kwargs.pop('activation')
        self._project = Layer(self._out_size, 1, name='project', **conv_kwargs)
        if self._strides == 1:
            self._shortcut = Layer(self._out_size, 1, name='shortcut' **conv_kwargs) \
                if in_size != self._out_size else tf.identity

    def call(self, x, training=False):
        y = x
        y = self._expand(y, training=training)
        y = self._depthwise(y, training=training)
        if self._reduction_ratio:
            y = self._se(y)
        y = self._project(y, training=training)
        if self._strides == 1:
            x = self._shortcut(x)
            x = x + y
        else:
            x = y
        return x


@register_cnn('mbse')
class MobileBottleneckSECNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 channels=[16, 32, 32],
                 expansion_ratio=[4, 4, 4],
                 reduction_ratio=1,
                 name='mbse', 
                 kernel_initializer='orthogonal',
                 out_size=256,
                 filter_multiplier=1,
                 **kwargs):
        super().__init__(name=name)
        assert len(channels) == len(expansion_ratio), f'{channels} vs {expansion_ratio}'
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)

        assert 'activation' in kwargs
        assert 'norm' in kwargs
        self._conv_layers = []
        print('MobileBottleneckSECNN kwargs:')
        for k, v in kwargs.items():
            print(k, v)
        for i, (t, c) in enumerate(zip(expansion_ratio, channels)):
            c *= filter_multiplier
            self._conv_layers += [
                InvertedResidualSE(2, c, t, name=f'sb{i}_{c}', **kwargs),
                InvertedResidualSE(1, c, t, reduction_ratio, name=f'inv_resse{i}_{c}_1', **kwargs),
                InvertedResidualSE(1, c, t, reduction_ratio, name=f'inv_resse{i}_{c}_2', **kwargs),
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