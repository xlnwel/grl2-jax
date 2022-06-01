from algo.zero.elements.nn import *
from nn.func import mlp, nn_registry

""" Source this file to register Networks """


@nn_registry.register('att')
class Attention(Module):
    def __init__(
        self,
        query=None, 
        key=None, 
        value=None, 
        name='attention',
    ):
        self._query_layer = mlp(**query, name=f'query') if query else None
        self._key_layer = mlp(**key, name=f'key') if key else None
        self._value_layer = mlp(**value, name=f'value') if value else None
        super().__init__(name=name)

    def call(self, q, k, v, mask=None):
        if self._query_layer is not None:
            q = self._query_layer(q)
        if self._key_layer is not None:
            k = self._key_layer(k)
        if self._value_layer is not None:
            v = self._value_layer(v)
        # softmax(QK^T)V
        dot_product = tf.einsum('bad,baod->bao', q, k)
        if mask is not None:
            dot_product *= mask
        weights = tf.nn.softmax(dot_product)
        x = tf.einsum('bao,baod->bad', weights, v)
        return x
