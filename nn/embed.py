import numpy as np
import tensorflow as tf

from core.module import Module
from nn.registry import nn_registry
from utility.tf_utils import assert_rank_and_shape_compatibility


@nn_registry.register('embed')
class Embedding(Module):
    def __init__(self, name='embed', **config):
        super().__init__(name=name)
        config = config.copy()
        self.input_dim = config.pop('input_dim')
        self.embed_size = config.pop('embed_size')
        self.input_length = config.pop('input_length')
        embeddings_initializer = config.pop('embeddings_initializer', 'uniform')

        self._embed = tf.keras.layers.Embedding(
            self.input_dim, 
            self.embed_size, 
            embeddings_initializer=embeddings_initializer, 
            input_length=self.input_length, 
            name=name
        )

    def call(
        self, 
        x, 
        tile: bool=False, 
        mask_out_self: bool=False, 
        flatten=False
    ):
        """
        Args:
            tile: if true we replicate the input's last dimension, 
                this yields a tensor of shape (B, A, A, D) given 
                the input/resulting embedding of shape (B, A)/(B, A, D) 
            mask_out_self: if true (and <tile> must be true), we 
                make (B, i, i, D) = 0 for i in range(A)
            flatten
        """
        assert len(x.shape) == 2 or len(x.shape) == 3, x.shape
        T = x.shape[1] if len(x.shape) == 3 else 0
        if T:
            x = tf.reshape(x, (-1, *x.shape[2:]))
        assert x.shape[1:] == (self.input_length,), x.shape

        x = self._embed(x)
        if tile:
            # same as 
            # x = tf.expand_dims(x, 1)
            # x = tf.tile(x, (1, self.input_length, 1, 1))
            # but approximately 3 times faster, tested on mac without tf.function
            idx = np.tile(
                np.arange(0, self.input_length, dtype=np.int32), 
                [self.input_length, 1]
            )
            x = tf.gather(x, idx, axis=-2)
        if mask_out_self:
            assert tile, 'tile must be true when mask_out_self is true'
            assert x.shape[1:] == (self.input_length, self.input_length, self.embed_size), x.shape
            mask = np.eye(self.input_length).reshape(
                1, self.input_length, self.input_length, 1).astype(np.int32)
            mask = mask + tf.zeros_like(x, dtype=tf.int32)
            mask = tf.cast(mask, tf.bool)
            assert_rank_and_shape_compatibility([mask, x])
            x = tf.where(mask, tf.zeros_like(x), x)
        if flatten:
            assert x.shape[1:] == (self.input_length, self.input_length, self.embed_size), x.shape
            x = tf.reshape(x, (-1, self.input_length, self.input_length * self.embed_size))

        if T:
            x = tf.reshape(x, (-1, T, self.input_length, x.shape[-1]))

        return x

    def embedding_vars(self):
        return self._embed.variables[0]
