import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class Noisy(Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.noise_sigma = kwargs.get('noisy_sigma', .4)

    def build(self, input_shape):
        super().build(input_shape)
        self.last_dim = input_shape[-1]
        self.noisy_w = self.add_weight(
            'noise_kernel',
            shape=(self.last_dim, self.units),
            initializer=tf.keras.initializers.glorot_normal(),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        if self.use_bias:
            self.noisy_b = self.add_weight(
                'noise_bias',
                shape=[self.units],
                initializer=tf.constant_initializer(self.noise_sigma / np.sqrt(self.units)),
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True)
        else:
            self.bias = None
        self.epsilon_w = tf.Variable(tf.random.truncated_normal([self.last_dim, self.units], stddev=self.noise_sigma),
                                     trainable=False, name='epsilon_w')
        self.epsilon_b = tf.Variable(tf.random.truncated_normal([self.units], stddev=self.noise_sigma),
                                     trainable=False, name='epsilon_b')

    def noisy_layer(self, inputs):
        return tf.matmul(inputs, self.noisy_w * self.epsilon_w) + self.noisy_b * self.epsilon_b

    def call(self, inputs, reset=True, noisy=True):
        y = super().call(inputs)
        if reset:
            self.reset()
        if noisy:
            noise = self.noisy_layer(inputs)
            return y + noisy
        else:
            return y

    def reset(self):
        self.epsilon_w.assign(tf.random.truncated_normal([self.last_dim, self.units], stddev=self.noise_sigma))
        self.epsilon_b.assign(tf.random.truncated_normal([self.units], stddev=self.noise_sigma))