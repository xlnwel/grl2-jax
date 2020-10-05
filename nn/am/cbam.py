from tensorflow.keras import layers

from core.module import Module
from nn.registry import am_registry
from nn.utils import *


@am_registry.register('cbam')
class CBAM(Module):
    def __init__(self, 
                 ratio=1,
                 kernel_size=7,
                 ca_on=True,
                 sa_on=True,
                 out_activation='sigmoid',
                 name='cbam', 
                 **kwargs):
        super().__init__(name=name)
        self._ratio = ratio   # the inverse of the reduction ratio
        self._kernel_size = kernel_size
        self._ca_on = ca_on
        self._sa_on = sa_on
        self._out_activation = out_activation
        self._kwargs = kwargs

    def build(self, input_shape):
        kwargs = self._kwargs.copy()    # we cannot modify attribute of the layer in build, which will emit an error when save the model
        kernel_initializer = kwargs.get('kernel_initializer', 'glorot_uniform')
        filters = input_shape[-1]

        if self._ca_on:
            self._c_avg, self._c_max, self._c_concat, \
                self._c_exc, self._c_mul = \
                self._channel_attention(filters, kernel_initializer)
        if self._sa_on:
            self._s_avg, self._s_max, self._s_concat, \
                self._s_exc, self._s_mul = \
                self._spatial_attention(filters, self._kernel_size, kernel_initializer)
    
    def _channel_attention(self, filters, kernel_initializer):
        name_fn = lambda name: f'{self.scope_name}/channel/{name}'
        avg_squeeze = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True), 
            name=name_fn('avg_squeeze'))
        max_squeeze = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True), 
                name=name_fn('max_squeeze'))

        reduced_filters = max(int(filters * self._ratio), 1)
        concat = layers.Concatenate(axis=-1, name=name_fn('concat'))
        excitation = [
            layers.Dense(reduced_filters, 
                kernel_initializer=kernel_initializer, activation='relu',
                name=name_fn('reduce')),
            layers.Dense(filters,
                activation=self._out_activation,
                name=name_fn('expand'))
        ]
        mul = layers.Multiply(name=name_fn('mul'))

        return avg_squeeze, max_squeeze, concat, excitation, mul

    def _spatial_attention(self, filters, kernel_size, kernel_initializer):
        name_fn = lambda name: f'{self.scope_name}/spatial/{name}'
        avg_squeeze = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), 
            name=name_fn('avg_squeeze'))
            
        max_squeeze = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True), 
                name=name_fn('max_squeeze'))

        concat = layers.Concatenate(axis=-1, name=name_fn('concat'))
        excitation = layers.Conv2D(1, kernel_size, 
                            strides=1, padding='same',
                            kernel_initializer=kernel_initializer, 
                            activation=self._out_activation,
                            use_bias=False,
                            name=name_fn('excitation'))
        mul = layers.Multiply(name=name_fn('mul'))
        return avg_squeeze, max_squeeze, concat, excitation, mul


    def call(self, x, **kwargs):
        if self._ca_on:
            c_avg, c_max = x, x
            c_avg = self._c_avg(c_avg)
            c_max = self._c_max(c_max)
            y = self._c_concat([c_avg, c_max])
            for l in self._c_exc:
                y = l(y)
            x = self._c_mul([x, y])
        
        if self._sa_on:
            s_avg = self._s_avg(x)
            s_max = self._s_max(x)
            y = self._s_concat([s_avg, s_max])
            y = self._s_exc(y)
            x = self._s_mul([x, y])
        
        return x

if __name__ == "__main__":
    for sa in [True, False]:
        se = CBAM(2, name='scope/cbam', sa_on=sa)
        x = tf.keras.layers.Input(shape=(64, 64, 12))
        y = se(x)
        m = tf.keras.Model(x, y)
        m.summary()
