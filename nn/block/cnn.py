"""
This file defines general CNN architectures used in RL
"""

import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPooling2D, TimeDistributed
from tensorflow.keras.activations import relu
from tensorflow.keras.mixed_precision.experimental import global_policy


mapping = dict(none=None)

def cnn(name, **kwargs):
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

def convert_obs(x, obs_range, dtype):
    if x.dtype != np.uint8:
        print(f'Observations are already converted to {x.dtype}, no further process is performed')
        return x
    assert x.dtype == np.uint8, x.dtype
    if obs_range == [0, 1]:
        return tf.cast(x, dtype) / 255.
    elif obs_range == [-.5, .5]:
        return tf.cast(x, dtype) / 255. - .5
    elif obs_range == [-1, 1]:
        return tf.cast(x, dtype) / 127.5 - 1.
    else:
        raise ValueError(obs_range)


@register('ftw')
class FTWCNN(Layer):
    def __init__(self, *, time_distributed=False, name='ftw', obs_range=[0, 1], **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        self._conv1 = conv2d(32, 8, strides=4, padding='same', **kwargs)
        self._conv2 = conv2d(64, 4, strides=2, padding='same', **kwargs)
        self._conv3 = conv2d(64, 3, strides=1, padding='same', **kwargs)
        self._conv4 = conv2d(64, 3, strides=1, padding='same', **kwargs)

        self.out_size = 256
        self._dense = Dense(self.out_size, activation=relu)

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
        shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
        x = tf.reshape(x, shape)
        x = self._dense(x)

        return x


class Residual(Layer):
    def __init__(self, time_distributed=False, name=None, **kwargs):
        super().__init__(name=name)
        self._time_distributed = time_distributed
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)
        conv2d = functools.partial(conv2d_fn, time_distributed=self._time_distributed)
        filters = input_shape[-1]
        
        self._conv1 = conv2d(filters, 3, strides=1, padding='same', **self._kwargs)
        self._conv2 = conv2d(filters, 3, strides=1, padding='same', **self._kwargs)

    def call(self, x):
        y = relu(x)
        y = self._conv1(y)
        y = relu(y)
        y = self._conv2(y)
        return x + y


@register('impala')
class IMPALACNN(Layer):
    def __init__(self, *, time_distributed=False, obs_range=[0, 1], name='impala', **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        maxpooling2d = lambda *args, **kwargs: (
            TimeDistributed(MaxPooling2D(*args, **kwargs))
            if time_distributed else
            MaxPooling2D(*args, **kwargs)
        )

        self._conv_layers = []
        for filters in [16, 32, 32]:
            self._conv_layers += [
                conv2d(filters, 3, strides=1, padding='same', **kwargs),
                maxpooling2d(3, strides=2, padding='same'),
                Residual(time_distributed=time_distributed, **kwargs),
                Residual(time_distributed=time_distributed, **kwargs),
            ]

        self.out_size = 256
        self._dense = Dense(self.out_size, activation=relu)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
        x = tf.reshape(x, shape)
        x = relu(x)
        x = self._dense(x)

        return x


@register('nature')
class NatureCNN(Layer):
    def __init__(self, *, time_distributed=False, obs_range=[0, 1], name='nature', **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        conv2d = functools.partial(conv2d_fn, time_distributed=time_distributed)
        self._conv_layers = [
            conv2d(32, 8, 4, padding='same', activation=relu),
            conv2d(64, 4, 2, padding='same', activation=relu),
            conv2d(64, 3, 1, padding='same', activation=relu),
        ]
        self.out_size = 512
        self._dense = Dense(self.out_size, activation=relu)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
        x = tf.reshape(x, shape)
        x = self._dense(x)
        
        return x
