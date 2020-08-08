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

Layer = layers.Layer
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

maxpooling2d = functools.partial(time_dist_fn, layers.MaxPooling2D)


""" FTW CNN """
@register('ftw')
class FTWCNN(Layer):
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
class Residual(Layer):
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
class IMPALACNN(Layer):
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
class SE(Layer):
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


class ResidualSE(Layer):
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
class ImpalaSECNN(Layer):
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


@register('nature')
class NatureCNN(Layer):
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
class SimpleCNN(Layer):
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

