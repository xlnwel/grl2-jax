"""
This file defines general CNN architectures used in RL
All CNNs eventually return a tensor of shape `[batch_size, n_features]`
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu

Conv2D = layers.Conv2D


def get_cnn(name):
    if name.lower() == 'ftw':
        return FTWCNN()
    elif name.lower() == 'impala':
        return IMPALACNN()
    else:
        raise NotImplementedError(f'Unknown CNN structure: {name}')

class FTWCNN(layers.Layer):
    def __init__(self, name='ftw'):
        super().__init__(name=name)
        self.conv1 = Conv2D(32, 8, strides=4, padding='same')
        self.conv2 = Conv2D(64, 4, strides=2, padding='same')
        self.conv3 = Conv2D(64, 3, strides=1, padding='same')
        self.conv4 = Conv2D(64, 3, strides=1, padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256)

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
        x = self.flatten(x)
        x = self.dense(x)
        x = relu(x)

        return x


class IMPALAResidual(layers.Layer):
    def __init__(self, filters, name=None):
        super().__init__(name=name)
        self.conv1 = Conv2D(filters, 3, strides=1, padding='same')
        self.conv2 = Conv2D(filters, 3, strides=1, padding='same')
    
    def call(self, x):
        y = relu(x)
        y = self.conv1(y)
        y = relu(y)
        y = self.conv2(y)
        return x + y


class IMPALACNN(layers.Layer):
    def __init__(self, name='impala'):
        super().__init__(name=name)
        self.cnn_layers = []
        for filters in [16, 32, 32]:
            self.cnn_layers += [
                Conv2D(filters, 3, strides=1, padding='same'),
                layers.MaxPooling2D(3, strides=2, padding='same'),
                IMPALAResidual(filters),
                IMPALAResidual(filters),
            ]
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256)
    
    def call(self, x):
        for l in self.cnn_layers:
            x = l(x)
        x = relu(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = relu(x)

        return x