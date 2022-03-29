from algo.hm.elements.nn import *
from utility.tf_utils import assert_rank_and_shape_compatibility
from utility.utils import positional_encodings


@nn_registry.register('ae')
class ActionEmbed(Module):
    def __init__(self, name='action_embed', **config):
        super().__init__(name=name)
        config = config.copy()
        self.embed_size = config.pop('embed_size')
        self.action_dim = config.pop('action_dim')
        self.n_units = config.pop('n_units')

        self._embed = tf.keras.layers.Embedding(
            self.action_dim, self.embed_size, 
            input_length=self.n_units, name='ae'
        )

        self._layers = mlp(
            **config, 
            out_dtype='float32',
            name=name
        )

    def call(self, action):
        assert len(action.shape) == 2 or len(action.shape) == 3, action.shape
        T = action.shape[1] if len(action.shape) == 3 else 0
        if T:
            action = tf.reshape(action, (-1, *action[2:]))
        assert action.shape[1:] == (self.n_units,), action.shape
        # uid = positional_encodings(self.n_units, self.embed_size)
        # assert uid.shape == (self.n_units, self.embed_size), uid.shape
        # embed = self._embed(action)
        # assert embed.shape[1:] == (self.n_units, self.embed_size), embed.shape
        # idx = np.stack([
        #     np.concatenate([
        #         np.arange(i+1, self.n_units, dtype=np.int32),
        #         np.arange(0, i, dtype=np.int32), 
        #     ], -1)
        #     for i in range(self.n_units)
        # ])
        # uid = uid[idx]
        # embed = tf.gather(embed, idx, axis=-2)
        # assert uid.shape == (self.n_units, (self.n_units-1), self.embed_size), uid.shape
        # assert embed.shape[1:] == (self.n_units, (self.n_units-1), self.embed_size), embed.shape
        # x = uid + embed
        # assert x.shape[1:] == (self.n_units, (self.n_units-1), self.embed_size), x.shape
        # x = tf.reshape(x, (-1, self.n_units, (self.n_units-1) * self.embed_size))
        # assert x.shape[1:] == (self.n_units, (self.n_units-1) * self.embed_size), x.shape
        
        idx = np.tile(np.arange(0, self.n_units, dtype=np.int32), [self.n_units, 1])
        x = self._embed(action)
        x = tf.gather(x, idx, axis=-2)
        # uid = uid[idx]
        # assert uid.shape == (self.n_units, self.n_units, self.embed_size), uid.shape
        # x = x + uid
        assert x.shape[1:] == (self.n_units, self.n_units, self.embed_size), x.shape
        mask = np.eye(self.n_units).reshape(1, self.n_units, self.n_units, 1).astype(np.int32)
        mask = mask + tf.zeros_like(x, dtype=tf.int32)
        mask = tf.cast(mask, tf.bool)
        # mask = np.tile(mask, [1, 1, self.embed_size])
        # mask = tf.tile(mask, [x.shape[0], 1, 1, 1])
        assert_rank_and_shape_compatibility([mask, x])
        x = tf.where(mask, tf.zeros_like(x), x)
        assert x.shape[1:] == (self.n_units, self.n_units, self.embed_size), x.shape
        x = tf.reshape(x, (-1, self.n_units, self.n_units * self.embed_size))

        x = self._layers(x)

        if T:
            x = tf.reshape(x, (-1, T, self.n_units, x.shape[-1]))

        return x
