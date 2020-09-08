from nn.utils import *
from nn.block.cnns.utils import *
from nn.block.cnns.se import SE


def _call_norm(norm_type, norm_layer, x, training):
    if norm_type == 'batch':
        y = norm_layer(x, training=training)
    else:
        y = norm_layer(x)
    return y


class ResidualV1(layers.Layer):
    def __init__(self, 
                 name='resv1', 
                 filters=[],
                 kernel_sizes=[3, 3],
                 activation='relu',
                 norm=None,
                 norm_kwargs={},
                 se_kwargs={},
                 **kwargs):
        super().__init__(name=name)
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._activation = get_activation(activation)
        self._norm = norm
        self._norm_kwargs = norm_kwargs
        self._se_kwargs = se_kwargs
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)

        kwargs = self._kwargs.copy()
        filters = self._filters or [input_shape[-1] for _ in self._kernel_sizes]
        se_kwargs = self._se_kwargs.copy()
        se_kwargs.update(kwargs)
        
        self._layers = []
        self._norm_class = get_norm(self._norm) if self._norm else None
        assert len(filters) == len(self._kernel_sizes), (filters, self._kernel_sizes)
        for f, ks in zip(filters, self._kernel_sizes):
            self._layers.append(conv2d(f, ks, strides=1, padding='same', **kwargs))
            if self._norm:
                self._layers.append(self._norm_class(**self._norm_kwargs))
            else:
                self._layers.append(lambda x: x)
            self._layers.append(self._activation)
        if self._se_kwargs:
            self._se = SE(**se_kwargs)

    def call(self, x, training=True):
        def call_norm(norm_layer, x):
            return _call_norm(self._norm, norm_layer, x, training)
        y = x
        for l in self._layers:
            if self._norm and isinstance(l, self._norm_class):
                y = call_norm(l, y)
            else:
                y = l(y)

        if self._se_kwargs:
            y = self._se(y)
        return x + y


class ResidualV2(layers.Layer):
    def __init__(self, 
                 name='resv2', 
                 filters=[],
                 kernel_sizes=[3, 3],
                 activation='relu',
                 norm=None,
                 norm_kwargs={},
                 se_kwargs={},
                 rezero=False,
                 **kwargs):
        super().__init__(name=name)
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._activation = get_activation(activation)
        self._norm = norm
        self._norm_kwargs = norm_kwargs
        self._se_kwargs = se_kwargs
        self._rezero = rezero
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)

        kwargs = self._kwargs.copy()
        filters = self._filters or [input_shape[-1] for _ in self._kernel_sizes]
        se_kwargs = self._se_kwargs.copy()
        se_kwargs.update(kwargs)
        
        self._layers = []
        self._norm_class = get_norm(self._norm) if self._norm else None
        assert len(filters) == len(self._kernel_sizes), (filters, self._kernel_sizes)
        for f, ks in zip(filters, self._kernel_sizes):
            if self._norm:
                self._layers.append(self._norm_class(**self._norm_kwargs))
            else:
                self._layers.append(lambda x: x)
            self._layers.append(self._activation)
            self._layers.append(conv2d(f, ks, strides=1, padding='same', **kwargs))
        if self._se_kwargs:
            self._se = SE(**se_kwargs)
        if self._rezero:
            self._rezero = tf.Variable(0., trainable=True, dtype=tf.floag32, name='rezero')

    def call(self, x, training=True):
        def call_norm(norm_layer, x):
            return _call_norm(self._norm, norm_layer, x, training)
        y = x
        for l in self._layers:
            if self._norm and isinstance(l, self._norm_class):
                y = call_norm(l, y)
            else:
                y = l(y)

        if self._se_kwargs:
            y = self._se(y)
        if self._rezero:
            y = self._rezero * y
        return x + y
