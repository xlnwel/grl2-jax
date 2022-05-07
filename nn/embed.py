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
        multiply: bool=False, 
        tile: bool=False, 
        mask_out_self: bool=False, 
        flatten=False,
        mask=None,
        time_distributed=False, 
    ):
        """
        Args:
            tile: If true we replicate the input's last dimension, 
                this yields a tensor of shape (B, U, U, D) given 
                the input/resulting embedding of shape (B, U)/(B, U, D).
                This is useful in MARL, where we consider other agents'
                actions for the current agent.
            mask_out_self: If true (and <tile> must be true), we 
                make (B, i, i, D) = 0 for i in range(A)
            flatten: Flatten the tiled tensor.
        """
        if time_distributed:
            T = x.shape[1]
            x = tf.reshape(x, (-1, *x.shape[2:]))
        if multiply:
            assert x.shape[1:] == (self.input_length, self.input_dim), \
                (x.shape, (self.input_length, self.input_dim))
            x = x @ self._embed.variables[0]
            assert x.shape[1:] == (self.input_length, self.embed_size), \
                (x.shape, (self.input_length, self.embed_size))
        else:
            assert x.shape[1:] == (self.input_length,), (x.shape, self.input_length)

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
            if mask is not None:
                mask = tf.cast(mask, tf.bool)
                mask1 = mask[..., None, :, None]
                mask2 = mask[..., None, None]
                mask = tf.math.logical_and(mask1, mask2)
                x = tf.where(mask, x, tf.zeros_like(x))
                tf.debugging.assert_equal(
                    tf.where(mask1, tf.zeros_like(x), x), 0.)
                tf.debugging.assert_equal(
                    tf.where(mask2, tf.zeros_like(x), x), 0.)
        if mask_out_self:
            assert tile, 'tile must be true when mask_out_self is true'
            assert x.shape[1:] == (self.input_length, self.input_length, self.embed_size), x.shape
            self_mask = np.eye(self.input_length).reshape(
                1, self.input_length, self.input_length, 1).astype(np.int32)
            self_mask = self_mask + tf.zeros_like(x, dtype=tf.int32)
            self_mask = tf.cast(self_mask, tf.bool)
            assert_rank_and_shape_compatibility([self_mask, x])
            x = tf.where(self_mask, tf.zeros_like(x), x)
        if flatten:
            assert x.shape[1:] == (self.input_length, self.input_length, self.embed_size), x.shape
            x = tf.reshape(x, (-1, self.input_length, self.input_length * self.embed_size))

        if time_distributed:
            x = tf.reshape(x, (-1, T, self.input_length, *x.shape[2:]))

        return x

    def embedding_vars(self):
        return self._embed.variables[0]
