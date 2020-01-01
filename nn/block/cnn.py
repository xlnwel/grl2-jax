"""
This file defines general CNN architectures used in RL
All CNNs eventually return a tensor of shape `[batch_size, n_features]`
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPooling2D, TimeDistributed
from tensorflow.keras.activations import relu


class FTWCNN(Layer):
    def __init__(self, *, time_distributed=False, batch_size=None, name='ftw'):
        super().__init__(name=name)
        conv2d = lambda *args, **kwargs: (
            TimeDistributed(Conv2D(*args, **kwargs))
            if time_distributed else
            Conv2D(*args, **kwargs)
        )
        self.conv1 = conv2d(32, 8, strides=4, padding='same')
        self.conv2 = conv2d(64, 4, strides=2, padding='same')
        self.conv3 = conv2d(64, 3, strides=1, padding='same')
        self.conv4 = conv2d(64, 3, strides=1, padding='same')

        self.out_size = 256
        self.dense = Dense(self.out_size)

        if time_distributed:
            assert batch_size is not None
            self.batch_size = batch_size

    def call(self, x):
        x = relu(self.conv1(x))
        x = self.conv2(x)
        y = relu(x)
        y = self.conv3(y)
        x = x + y
        y = relu(x)
        y = self.conv4(y)
        x = x + y
        x = relu(x)
        if isinstance(self.conv1, TimeDistributed):
            x = tf.reshape(x, (self.batch_size, -1, tf.reduce_prod(x.shape[2:])))
        else:
            x = tf.reshape(x, (self.batch_size, tf.reduce_prod(x.shape[1:])))
        x = self.dense(x)
        x = relu(x)

        return x


class IMPALAResidual(Layer):
    def __init__(self, filters, time_distributed=False, name=None):
        super().__init__(name=name)
        conv2d = lambda *args, **kwargs: (
            TimeDistributed(Conv2D(*args, **kwargs))
            if time_distributed else
            Conv2D(*args, **kwargs)
        )
        self.conv1 = conv2d(filters, 3, strides=1, padding='same')
        self.conv2 = conv2d(filters, 3, strides=1, padding='same')

    def call(self, x):
        y = relu(x)
        y = self.conv1(y)
        y = relu(y)
        y = self.conv2(y)
        return x + y


class IMPALACNN(Layer):
    def __init__(self, *, time_distributed=False, batch_size=None, name='impala'):
        super().__init__(name=name)
        conv2d = lambda *args, **kwargs: (
            TimeDistributed(Conv2D(*args, **kwargs))
            if time_distributed else
            Conv2D(*args, **kwargs)
        )
        maxpooling2d = lambda *args, **kwargs: (
            TimeDistributed(MaxPooling2D(*args, **kwargs))
            if time_distributed else
            MaxPooling2D(*args, **kwargs)
        )

        self.cnn_layers = []
        for filters in [16, 32, 32]:
            self.cnn_layers += [
                conv2d(filters, 3, strides=1, padding='same'),
                maxpooling2d(3, strides=2, padding='same'),
                IMPALAResidual(filters, time_distributed=time_distributed),
                IMPALAResidual(filters, time_distributed=time_distributed),
            ]

        self.out_size = 256
        self.dense = Dense(self.out_size)

        if time_distributed:
            assert batch_size is not None
            self.batch_size = batch_size
    
    def call(self, x):
        for l in self.cnn_layers:
            x = l(x)
        x = relu(x)
        if isinstance(self.conv1, TimeDistributed):
            x = tf.reshape(x, (self.batch_size, -1, tf.reduce_prod(x.shape[2:])))
        else:
            x = tf.reshape(x, (self.batch_size, tf.reduce_prod(x.shape[1:])))
        x = self.dense(x)
        x = relu(x)

        return x
