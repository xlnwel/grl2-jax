import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from nn.utils import get_norm, get_activation, get_initializer


class Layer(layers.Layer):
    def __init__(self, units, layer_type=layers.Dense, norm=None, 
                activation=None, kernel_initializer='he_uniform', 
                name=None, **kwargs):
        super().__init__(name=name)

        norm=get_norm(norm)
        activation=get_activation(activation)
        kernel_initializer = get_initializer(kernel_initializer)

        self.intra_layers = [layer_type(units, kernel_initializer=kernel_initializer, **kwargs)]
        if norm is not None:
            self.intra_layers.append(norm())

        self.activation=activation
    
    def call(self, x, **kwargs):
        for l in self.intra_layers:
            x = l(x, **kwargs)
        if self.activation is not None:
            x = self.activation(x)
        
        return x
    
    def reset(self):
        for l in self.intra_layers:
            # reset noisy layer
            if isinstance(l, Noisy):
                l.reset()

class Noisy(layers.Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.noise_sigma = kwargs.get('noisy_sigma', .4)

    def build(self, input_shape):
        super().build(input_shape)
        self.last_dim = input_shape[-1]
        self.noisy_w = self.add_weight(
            'noise_kernel',
            shape=(self.last_dim, self.units),
            initializer=get_initializer('he_normal'),
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
        self.epsilon_w_in = tf.Variable(
            tf.random.truncated_normal([self.last_dim, 1], stddev=self.noise_sigma),
                                        trainable=False, name='epsilon_w_in')
        self.epsilon_w_out = tf.Variable(
            tf.random.truncated_normal([1, self.units], stddev=self.noise_sigma),
                                        trainable=False, name='epsilon_w_out')
        self.epsilon_b = tf.Variable(
            tf.random.truncated_normal([self.units], stddev=self.noise_sigma),
                                        trainable=False, name='epsilon_b')

    def noisy_layer(self, inputs):
        epsilon_w_in = tf.math.sign(self.epsilon_w_in) * tf.math.sqrt(tf.math.abs(self.epsilon_w_in))
        epsilon_w_out = tf.math.sign(self.epsilon_w_out) * tf.math.sqrt(tf.math.abs(self.epsilon_w_out))
        epsilon_w = tf.matmul(epsilon_w_in, epsilon_w_out)
        return tf.matmul(inputs, self.noisy_w * epsilon_w) + self.noisy_b * self.epsilon_b

    def call(self, inputs, reset=True, noisy=True):
        y = super().call(inputs)
        if noisy:
            if reset:
                self.reset()
            noise = self.noisy_layer(inputs)
            y = y + noise
        return y

    def det_step(self, inputs):
        return super().call(inputs)

    def reset(self):
        self.epsilon_w_in.assign(tf.random.truncated_normal([self.last_dim, 1], stddev=self.noise_sigma))
        self.epsilon_w_out.assign(tf.random.truncated_normal([1, self.units], stddev=self.noise_sigma))
        self.epsilon_b.assign(tf.random.truncated_normal([self.units], stddev=self.noise_sigma))