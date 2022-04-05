import numpy as np
import tensorflow as tf

from core.module import Module
from nn.registry import nn_registry
from nn.mlp import MLP
from utility.tf_utils import assert_rank_and_shape_compatibility


@nn_registry.register('embed')
class Embedding(Module):
    def __init__(self, name='action_embed', **config):
        super().__init__(name=name)
        config = config.copy()
        self.input_dim = config.pop('input_dim')
        self.embed_size = config.pop('embed_size')
        self.input_length = config.pop('input_length')

        self._embed = tf.keras.layers.Embedding(
            self.input_dim, 
            self.embed_size, 
            input_length=self.input_length, 
            name=f'{name}/ae'
        )

        self._layers = MLP(
            **config, 
            out_dtype='float32',
            name=name
        )

    def call(self, x):
        assert len(x.shape) == 2 or len(x.shape) == 3, x.shape
        T = x.shape[1] if len(x.shape) == 3 else 0
        if T:
            x = tf.reshape(x, (-1, *x.shape[2:]))
        assert x.shape[1:] == (self.input_length,), x.shape

        idx = np.tile(np.arange(0, self.input_length, dtype=np.int32), [self.input_length, 1])
        x = self._embed(x)
        x = tf.gather(x, idx, axis=-2)
        assert x.shape[1:] == (self.input_length, self.input_length, self.embed_size), x.shape
        mask = np.eye(self.input_length).reshape(1, self.input_length, self.input_length, 1).astype(np.int32)
        mask = mask + tf.zeros_like(x, dtype=tf.int32)
        mask = tf.cast(mask, tf.bool)
        assert_rank_and_shape_compatibility([mask, x])
        x = tf.where(mask, tf.zeros_like(x), x)
        assert x.shape[1:] == (self.input_length, self.input_length, self.embed_size), x.shape
        x = tf.reshape(x, (-1, self.input_length, self.input_length * self.embed_size))

        x = self._layers(x)

        if T:
            x = tf.reshape(x, (-1, T, self.input_length, x.shape[-1]))

        return x

    def embedding_vars(self):
        return self._embed.variables[0]
