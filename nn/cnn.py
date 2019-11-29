"""
This file defines general CNN architectures used in RL
All CNNs eventually return a tensor of shape `[batch_size, n_features]`
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu


def get_cnn(name):
    if name.lower() == 'ftw':
        return FTWCNN()
    else:
        raise NotImplementedError('Unknown CNN structure: {name}')

class FTWCNN(layers.Layer):
    def __init__(self, name='ftw'):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(32, 8, strides=4, padding='same')
        self.conv2 = layers.Conv2D(64, 4, strides=2, padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=1, padding='same')
        self.conv4 = layers.Conv2D(64, 3, strides=1, padding='same')
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

        return x
