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
            self._c_avg, self._c_max, self._c_add, self._c_act, self._c_mul = \
                self._channel_attention(filters, kernel_initializer)
        if self._sa_on:
            self._s_avg, self._s_max, self._s_concat, self._s_excitation, self._s_mul = \
                self._spatial_attention(filters, self._kernel_size, kernel_initializer)
    
    def _channel_attention(self, filters, kernel_initializer):
        name_fn = lambda name: f'{self.scope_name}/channel/{name}'
        avg_squeeze = [
            layers.GlobalAvgPool2D(name=name_fn('avg_squeeze')),
            layers.Reshape((1, 1, filters), name=name_fn('avg_reshape')),
        ]
        max_squeeze = [
            layers.GlobalMaxPool2D(name=name_fn('max_squeeze')),
            layers.Reshape((1, 1, filters), name=name_fn('max_reshape')),
        ]

        # TODO: Use different excitation for avg and max
        reduced_filters = max(int(filters * self._ratio), 1)
        avg_excitation = [
            layers.Dense(reduced_filters, 
                kernel_initializer=kernel_initializer, activation='relu',
                name=name_fn('avg_reduce')),
            layers.Dense(filters,
                name=name_fn('avg_expand'))
        ]
        max_excitation = [
            layers.Dense(reduced_filters, 
                kernel_initializer=kernel_initializer, activation='relu',
                name=name_fn('max_reduce')),
            layers.Dense(filters,
                name=name_fn('max_expand'))
        ]
        add = layers.Add(name=name_fn('add'))
        act = get_activation(self._out_activation, return_cls=True)(
            name=name_fn(self._out_activation))
        mul = layers.Multiply(name=name_fn('mul'))

        return avg_squeeze + avg_excitation, max_squeeze + max_excitation, add, act, mul

    def _spatial_attention(self, filters, kernel_size, kernel_initializer):
        name_fn = lambda name: f'{self.scope_name}/spatial/{name}'
        avg_squeeze = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), 
            name=name_fn('avg_squeeze'))
            
        max_squeeze = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True), 
                name=name_fn('max_squeeze'))

        concat = layers.Concatenate(axis=-1, name=f'{self.scope_name}')
        excitation = layers.Conv2D(1, kernel_size, 
                            strides=1, padding='same',
                            kernel_initializer=kernel_initializer, 
                            activation=self._out_activation,
                            use_bias=False,
                            name=name_fn('excitation'))
        mul = layers.Multiply(name=name_fn('mul'))
        return avg_squeeze, max_squeeze, concat, excitation, mul


    def call(self, x):
        if self._ca_on:
            c_avg, c_max = x, x
            for l in self._c_avg:
                c_avg = l(c_avg)
            for l in self._c_max:
                c_max = l(c_max)
            y = self._c_add([c_avg, c_max])
            y = self._c_act(y)
            x = self._c_mul([x, y])
        
        if self._sa_on:
            s_avg = self._s_avg(x)
            s_max = self._s_max(x)
            y = self._s_concat([s_avg, s_max])
            y = self._s_excitation(y)
            x = self._s_mul([x, y])
        
        return x

if __name__ == "__main__":
    for td in [True, False]:
        shape = (4, 3, 3, 2) if td else (3, 3, 2)
        se = CBAM(2, name='scope/cbam')
        x = tf.keras.layers.Input(shape=shape)
        y = se(x)
        m = tf.keras.Model(x, y)
        m.summary()
