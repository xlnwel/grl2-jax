"""
This file defines general CNN architectures used in RL
"""

import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.mixed_precision.experimental import global_policy

from nn.utils import *
from nn.layers import Layer

Dense = layers.Dense

mapping = dict(none=None)

def cnn(name, **kwargs):
    if name is None:
        return None
    name = name.lower()
    if name in mapping:
        return mapping[name](**kwargs)
    else:
        raise ValueError(f'Unknown CNN structure: {name}')

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


time_dist_fn = lambda fn, *args, time_distributed=False, **kwargs: (
    layers.TimeDistributed(fn(*args, **kwargs))
    if time_distributed else
    fn(*args, **kwargs)
)

conv2d = functools.partial(time_dist_fn, layers.Conv2D)
depthwise_conv2d = functools.partial(time_dist_fn, layers.DepthwiseConv2D)

maxpooling2d = functools.partial(time_dist_fn, layers.MaxPooling2D)


""" FTW CNN """
@register('ftw')
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
            self._dense = Dense(self.out_size, activation=relu,
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


""" Impala CNN """
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


@register('impala')
class IMPALACNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 name='impala', 
                 kernel_initializer='orthogonal',
                 out_size=256,
                 filter_multiplier=1,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)

        self._conv_layers = []
        for i, filters in enumerate([16, 32, 32]):
            filters *= filter_multiplier
            self._conv_layers += [
                conv2d(filters, 3, strides=1, padding='same', **kwargs),
                maxpooling2d(3, strides=2, padding='same', time_distributed=time_distributed),
                Residual(name=f'res{i}_{filters}_1', **kwargs),
                Residual(name=f'res{i}_{filters}_2', **kwargs),
            ]

        self.out_size = out_size
        if self.out_size:
            self._dense = Dense(self.out_size, activation=relu)
    
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
            Dense(channels // self._reduction_ratio, kernel_initializer=kernel_initializer, activation='relu'),
            Dense(channels, activation='sigmoid')
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


@register('impalase')
class ImpalaSECNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 reduction_ratio=1,
                 name='impalase', 
                 kernel_initializer='orthogonal',
                 out_size=256,
                 filter_multiplier=1,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)

        self._conv_layers = []
        for i, filters in enumerate([16, 32, 32]):
            filters *= filter_multiplier
            self._conv_layers += [
                conv2d(filters, 3, strides=1, padding='same', **kwargs),
                maxpooling2d(3, strides=2, padding='same', time_distributed=time_distributed),
                ResidualSE(reduction_ratio, name=f'resse{i}_{filters}_1', **kwargs),
                ResidualSE(reduction_ratio, name=f'resse{i}_{filters}_2', **kwargs),
            ]

        self.out_size = out_size
        if self.out_size:
            self._dense = Dense(self.out_size, activation=relu)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = relu(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)

        return x


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
        conv_kwargs.pop('activation')
        self._project = Layer(self._out_size, 1, name='project', **conv_kwargs)
        if self._reduction_ratio:
            self._se = SE(self._reduction_ratio, name='se', **kwargs)
        if self._strides == 1:
            self._shortcut = Layer(self._out_size, 1, name='shortcut' **conv_kwargs) \
                if in_size != self._out_size else tf.identity

    def call(self, x, training=False):
        y = x
        y = self._expand(y, training=training)
        y = self._depthwise(y, training=training)
        y = self._project(y, training=training)
        if self._reduction_ratio:
            y = self._se(y)
        if self._strides == 1:
            x = self._shortcut(x)
            x = x + y
        else:
            x = y
        return x


@register('mbse')
class MobileBottleneckSECNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 reduction_ratio=1,
                 name='mbse', 
                 kernel_initializer='orthogonal',
                 out_size=256,
                 filter_multiplier=1,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)

        assert 'activation' in kwargs
        assert 'norm' in kwargs
        self._conv_layers = []
        expansion_ratios = [4, 4, 4]
        channels = [16, 32, 32]
        print('MobileBottleneckSECNN kwargs:')
        for k, v in kwargs.items():
            print(k, v)
        for i, (t, c) in enumerate(zip(expansion_ratios, channels)):
            c *= filter_multiplier
            self._conv_layers += [
                InvertedResidualSE(2, c, t, name=f'sb{i}_{c}', **kwargs),
                InvertedResidualSE(1, c, t, reduction_ratio, name=f'inv_resse{i}_{c}_1', **kwargs),
                InvertedResidualSE(1, c, t, reduction_ratio, name=f'inv_resse{i}_{c}_2', **kwargs),
            ]

        self.out_size = out_size
        if self.out_size:
            self._dense = Dense(self.out_size, activation=relu)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = relu(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)

        return x


@register('nature')
class NatureCNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 name='nature', 
                 kernel_initializer='orthogonal',
                 activation='relu',
                 out_size=512,
                 padding='valid',
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain(activation))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)
        activation = get_activation(activation)
        kwargs['activation'] = activation
        kwargs['padding'] = padding

        self._conv_layers = [
            conv2d(32, 8, 4, **kwargs),
            conv2d(64, 4, 2, **kwargs),
            conv2d(64, 3, 1, **kwargs),
        ]
        self.out_size = out_size
        if out_size:
            self._dense = Dense(self.out_size, activation=relu)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)
        
        return x

@register('simple')
class SimpleCNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 name='nature', 
                 kernel_initializer='orthogonal',
                 activation='relu',
                 out_size=256,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain(activation))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)
        activation = get_activation(activation)
        kwargs['activation'] = activation

        self._conv_layers = [
            conv2d(32, 5, 5, **kwargs),
            conv2d(64, 5, 5, **kwargs),
        ]
        self.out_size = out_size
        if out_size:
            self._dense = Dense(self.out_size, activation=activation)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)
        
        return x

