"""
This file defines general CNN architectures used in RL
"""

import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPooling2D, TimeDistributed
from tensorflow.keras.activations import relu
from tensorflow.keras.mixed_precision.experimental import global_policy

from nn.utils import get_initializer


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

def convert_obs(x, obs_range, dtype):
    if x.dtype != np.uint8:
        print(f'Observations are already converted to {x.dtype}, no further process is performed')
        return x
    assert x.dtype == np.uint8, x.dtype
    print(f'Observations are converted to range {obs_range} of dtype {dtype}')
    if obs_range == [0, 1]:
        return tf.cast(x, dtype) / 255.
    elif obs_range == [-.5, .5]:
        return tf.cast(x, dtype) / 255. - .5
    elif obs_range == [-1, 1]:
        return tf.cast(x, dtype) / 127.5 - 1.
    else:
        raise ValueError(obs_range)

def flatten(x):
    shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
    x = tf.reshape(x, shape)
    return x


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
        gain = kwargs.get('gain', np.sqrt(2))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)

        self._conv1 = conv2d(32, 8, strides=4, padding='same', 
                kernel_initializer=kernel_initializer, **kwargs)
        self._conv2 = conv2d(64, 4, strides=2, padding='same',
                kernel_initializer=kernel_initializer, **kwargs)
        self._conv3 = conv2d(64, 3, strides=1, padding='same', 
                kernel_initializer=kernel_initializer, **kwargs)
        self._conv4 = conv2d(64, 3, strides=1, padding='same', 
                kernel_initializer=kernel_initializer, **kwargs)

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


@register('r2d2')
class R2D2CNN(Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 name='ftw', 
                 obs_range=[0, 1], 
                 kernel_initializer='orthogonal',
                 out_size=512,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        gain = kwargs.get('gain', np.sqrt(2))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)

        self._conv1 = conv2d(32, 8, strides=4, padding='same', 
                kernel_initializer=kernel_initializer, **kwargs)
        self._conv2 = conv2d(64, 4, strides=2, padding='same',
                kernel_initializer=kernel_initializer, **kwargs)
        self._conv3 = conv2d(64, 3, strides=1, padding='same', 
                kernel_initializer=kernel_initializer, **kwargs)
        
        self.out_size = out_size
        if self.out_size:
            self._dense = Dense(self.out_size, activation=relu,
                            kernel_initializer=kernel_initializer, **kwargs)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        x = relu(self._conv1(x))
        x = relu(self._conv2(x))
        x = relu(self._conv3(x))
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
        self._kernel_initializer = kernel_initializer
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)
        conv2d = functools.partial(conv2d_fn, time_distributed=self._time_distributed)
        gain = self._kwargs.get('gain', np.sqrt(2))
        kernel_initializer = get_initializer(self._kernel_initializer, gain=gain)
        kwargs = self._kwargs
        filters = input_shape[-1]
        
        self._conv1 = conv2d(filters, 3, strides=1, padding='same', 
                            kernel_initializer=kernel_initializer, **kwargs)
        self._conv2 = conv2d(filters, 3, strides=1, padding='same', 
                            kernel_initializer=kernel_initializer, **kwargs)

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
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        maxpooling2d = functools.partial(maxpooling2d_fn, time_distributed=time_distributed)
        gain = kwargs.get('gain', np.sqrt(2))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)

        self._conv_layers = []
        for filters in [16, 32, 32]:
            self._conv_layers += [
                conv2d(filters, 3, strides=1, padding='same', 
                        kernel_initializer=kernel_initializer, **kwargs),
                maxpooling2d(3, strides=2, padding='same'),
                Residual(time_distributed=time_distributed, 
                        kernel_initializer=kernel_initializer, **kwargs),
                Residual(time_distributed=time_distributed, 
                        kernel_initializer=kernel_initializer, **kwargs),
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
                 out_size=512,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        gain = kwargs.get('gain', np.sqrt(2))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)
        
        self._conv_layers = [
            conv2d(32, 8, 4, padding='valid', activation=relu, 
                    kernel_initializer=kernel_initializer, **kwargs),
            conv2d(64, 4, 2, padding='valid', activation=relu, 
                    kernel_initializer=kernel_initializer, **kwargs),
            conv2d(64, 3, 1, padding='valid', activation=relu, 
                    kernel_initializer=kernel_initializer, **kwargs),
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
