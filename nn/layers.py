import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from nn.utils import get_norm, get_activation, get_initializer


class Layer(layers.Layer):
    def __init__(self, units, layer_type=layers.Dense, norm=None, 
                activation=None, kernel_initializer='glorot_uniform', 
                name=None, **kwargs):
        super().__init__(name=name)

        gain = np.sqrt(2) if activation == 'relu' and kernel_initializer == 'orthogonal' else 1.
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)

        self._layer = layer_type(units, kernel_initializer=kernel_initializer, **kwargs)
        self._norm_layer = get_norm(norm)
        if self._norm_layer:
            self._norm_layer = self._norm_layer()

        self.activation = get_activation(activation)

    def call(self, x, **kwargs):
        x = self._layer(x, **kwargs)
        if self._norm_layer:
            x = self._norm_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        
        return x
    
    def reset(self):
        # reset noisy layer
        if isinstance(self._layer, Noisy):
            self._layer.reset()

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
            initializer=get_initializer('glorot_normal'),
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
        self.eps_w_in = self.add_weight(
            'eps_w_in', 
            shape=(self.last_dim, 1),
            initializer=get_initializer('zeros'),
            trainable=False)
        self.eps_w_out = self.add_weight(
            'eps_w_out', 
            shape=(1, self.units),
            initializer=get_initializer('zeros'),
            trainable=False)
        self.eps_b = tf.reshape(self.eps_w_out, [self.units])

    def noisy_layer(self, inputs):
        eps_w_in = tf.math.sign(self.eps_w_in) * tf.math.sqrt(tf.math.abs(self.eps_w_in))
        eps_w_out = tf.math.sign(self.eps_w_out) * tf.math.sqrt(tf.math.abs(self.eps_w_out))
        eps_w = tf.matmul(eps_w_in, eps_w_out)
        return tf.matmul(inputs, self.noisy_w * eps_w) + self.noisy_b * self.eps_b

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
        self.eps_w_in.assign(tf.random.truncated_normal([self.last_dim, 1], stddev=self.noise_sigma))
        self.eps_w_out.assign(tf.random.truncated_normal([1, self.units], stddev=self.noise_sigma))
