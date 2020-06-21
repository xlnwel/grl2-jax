"""
This file defines general CNN architectures used in RL
"""

import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPooling2D, TimeDistributed
from tensorflow.keras.activations import relu
from tensorflow.keras.mixed_precision.experimental import global_policy

from nn.utils import *


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


conv2d_fn = lambda *args, time_distributed=False, **kwargs: (
    TimeDistributed(Conv2D(*args, **kwargs))
    if time_distributed else
    Conv2D(*args, **kwargs)
)

maxpooling2d_fn = lambda *args, time_distributed=False, **kwargs: (
    TimeDistributed(MaxPooling2D(*args, **kwargs))
    if time_distributed else
    MaxPooling2D(*args, **kwargs)
)


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

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
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
                            kernel_initializer=kernel_initializer, **kwargs)

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
        if self.out_size:
            x = flatten(x)
            x = self._dense(x)

        return x


class Residual(Layer):
    def __init__(self, 
                 time_distributed=False, 
                 name=None, 
                 kernel_initializer='orthogonal',
                 **kwargs):
        super().__init__(name=name)
        self._time_distributed = time_distributed
        kwargs.setdefault('kernel_initializer', kernel_initializer)
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)
        kwargs = self._kwargs
        conv2d = functools.partial(conv2d_fn, time_distributed=self._time_distributed)
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kernel_initializer = get_initializer(kwargs['kernel_initializer'], gain=gain)
        kwargs['kernel_initializer'] = kernel_initializer
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

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        maxpooling2d = functools.partial(maxpooling2d_fn, time_distributed=time_distributed)
        gain = kwargs.pop('gain', calculate_gain('relu'))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)
        kwargs['kernel_initializer'] = kernel_initializer

        self._conv_layers = []
        for filters in [16, 32, 32]:
            filters *= filter_multiplier
            self._conv_layers += [
                conv2d(filters, 3, strides=1, padding='same', **kwargs),
                maxpooling2d(3, strides=2, padding='same'),
                Residual(time_distributed=time_distributed, **kwargs),
                Residual(time_distributed=time_distributed, **kwargs),
            ]

        self.out_size = out_size
        if self.out_size:
            self._dense = Dense(self.out_size, activation=relu)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = relu(x)
        if self.out_size:
            x = flatten(x)
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
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        gain = kwargs.pop('gain', calculate_gain(activation))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)
        kwargs['kernel_initializer'] = kernel_initializer
        activation = get_activation(activation)
        kwargs['activation'] = activation

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
        if self.out_size:
            x = flatten(x)
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

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        gain = kwargs.pop('gain', calculate_gain(activation))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)
        kwargs['kernel_initializer'] = kernel_initializer
        activation = get_activation(activation)
        kwargs['activation'] = activation

        self._conv_layers = [
            conv2d(32, 5, 5, **kwargs),
            conv2d(64, 5, 5, **kwargs),
        ]
        self.out_size = out_size
        if out_size:
            self._dense = Dense(self.out_size, activation=relu)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        if self.out_size:
            x = flatten(x)
            x = self._dense(x)
        
        return x

