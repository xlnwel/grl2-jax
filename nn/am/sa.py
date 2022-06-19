import tensorflow as tf
from tensorflow.keras import layers

from core.module import Module
from nn.func import mlp
from nn.registry import layer_registry, block_registry, nn_registry
from nn.utils import get_norm, call_norm


def compute_dot_product_attention(q, k, v, mask=None, scale_logits=False):
    """ compute softmax(qk^T)v """
    if scale_logits:
        q *= q.shape[-1] ** -.5
    dot_product = tf.matmul(q, k, transpose_b=True)
    if mask is not None:
        dot_product *= mask
    weights = tf.nn.softmax(dot_product)
    x = tf.matmul(weights, v)
    return x

"""
Attention provides the most universal implementation of self-attention,
(and is probably the one would never be used), while Attention2 and 
Attention3 make simplifications to the input and neural network
"""
@block_registry.register('att')
@layer_registry.register('att')
class Attention(Module):
    def __init__(
        self,
        query=None, 
        key=None, 
        value=None, 
        scale_logits=False, 
        name='attention',
    ):
        self._query_layer = mlp(**query, name=f'query') if query else None
        self._key_layer = mlp(**key, name=f'key') if key else None
        self._value_layer = mlp(**value, name=f'value') if value else None
        self._scale_logits = scale_logits
        super().__init__(name=name)

    def call(self, q, k, v, mask=None):
        if self._query_layer is not None:
            q = self._query_layer(q)
        if self._key_layer is not None:
            k = self._key_layer(k)
        if self._value_layer is not None:
            v = self._value_layer(v)
        x = compute_dot_product_attention(
            q, k, v, 
            mask=mask, 
            scale_logits=self._scale_logits
        )
        return x


@block_registry.register('att2')
@layer_registry.register('att2')
class Attention2(Module):
    def __init__(
        self,
        key_size, 
        val_size, 
        scale_logits=False, 
        name='attention',
        **config, 
    ):
        """ Attention2 implemenets self-attention """
        self._key_size = key_size
        self._val_size = val_size
        self._scale_logits = scale_logits
        out_size = 2 * key_size + val_size
        self._layers = mlp(
            **config, 
            out_size=out_size, 
            name=f'{name}/embed'
        )
        super().__init__(name=name)

    def call(self, x, mask=None):
        qkv = self._layers(x)
        q, k, v = tf.split(qkv, [self._key_size, self._key_size, self._val_size], -1)
        x = compute_dot_product_attention(
            q, k, v, 
            mask=mask, 
            scale_logits=self._scale_logits
        )
        return x


@block_registry.register('att3')
@layer_registry.register('att3')
class Attention3(Module):
    def __init__(
        self,
        key_size, 
        val_size, 
        scale_logits=False, 
        name='attention',
        **config, 
    ):
        """ Attention3 distinguishes query from key and values but
        does not distinguish betwen key and value.
        """
        self._key_size = key_size
        self._val_size = val_size
        self._scale_logits = scale_logits
        self._query = mlp(
            **config, 
            out_size=key_size, 
            name=f'{name}/query_embed'
        )
        self._kv = mlp(
            **config, 
            out_size=key_size + val_size, 
            name=f'{name}/kv_embed'
        )
        super().__init__(name=name)

    def call(self, q, kv, mask=None):
        q = self._query(x)
        kv = self._kv(kv)
        k, v = tf.split(kv, [self._key_size, self._val_size], -1)
        x = compute_dot_product_attention(
            q, k, v, 
            mask=mask, 
            scale_logits=self._scale_logits
        )
        return x


@block_registry.register('mhsa')
@layer_registry.register('mhsa')
class MultiHeadSelfAttention(Module):
    def __init__(
        self,
        key_size,
        val_size,
        num_heads,
        scale_logits=True,
        out_size=None,
        drop_rate=0,
        name='mhsa',
        **mlp_config
    ):
        super().__init__(name=name)
        self._key_size = key_size
        self._val_size = val_size
        self._num_heads = num_heads
        self._scale_logits = scale_logits
        self._out_size = out_size
        self._drop_rate = drop_rate
        self._mlp_config = mlp_config

    def build(self, input_shape):
        assert len(input_shape) == 3, input_shape
        seqlen, out_size = input_shape[1:]
        qkv_size = 2 * self._key_size + self._val_size
        total_size = qkv_size * self._num_heads
        out_size = self._out_size or out_size

        prefix = f'{self.name}/'
        self._embed = mlp(
            **self._mlp_config, 
            out_size=total_size, 
            name=prefix+'embed'
        )

        self._group_heads = layers.Reshape((seqlen, self._num_heads, qkv_size), name=prefix+'group_heads')
        self._concat = layers.Reshape((seqlen, self._num_heads * self._val_size), name=prefix+'concat')
        self._out = mlp(
            **self._mlp_config, 
            out_size=out_size, 
            name=prefix+'out'
        )
        if self._drop_rate > 0:
            self._drop = layers.Dropout(self._drop_rate, (None, None, 1), name=prefix+'drop')
        
        super().build(input_shape)

    def call(self, x, training=False, mask=None):
        qkv = self._embed(x)
        qkv = self._group_heads(qkv)                            # [B, N, F] -> [B, N, H, F/H]
        qkv = tf.transpose(qkv, [0, 2, 1, 3])      # [B, N, H, F/H] -> [B, H, N, F/H]

        q, k, v = tf.split(qkv, [self._key_size, self._key_size, self._val_size], -1)
        
        # softmax(QK^T/(d**2))V
        out = compute_dot_product_attention(
            q, k, v, 
            mask=mask, 
            scale_logits=self._scale_logits
        )

        # [B, H, N, V] -> [B, N, H, V]
        out = tf.transpose(out, [0, 2, 1, 3])
        # [B, N, H, V] -> [B, N, H * V]
        x = self._concat(out)
        x = self._out(x)

        if self._drop_rate > 0:
            x = self._drop(x, training=training)

        return x


@block_registry.register('mhsa2')
@layer_registry.register('mhsa2')
class MultiHeadSelfAttention2(Module):
    def __init__(
        self,
        key_size,
        val_size,
        num_heads,
        scale_logits=True,
        out_size=None,
        drop_rate=0,
        name='mhsa',
        **mlp_config
    ):
        super().__init__(name=name)
        self._key_size = key_size
        self._val_size = val_size
        self._num_heads = num_heads
        self._scale_logits = scale_logits
        self._out_size = out_size
        self._drop_rate = drop_rate
        self._mlp_config = mlp_config

    def build(self, input_shape, *args):
        assert len(input_shape) == 3, input_shape
        seqlen, out_size = input_shape[1:]
        kv_size =self._key_size + self._val_size
        out_size = self._out_size or out_size

        prefix = f'{self.name}/'
        self._q_embed = mlp(
            **self._mlp_config, 
            out_size=self._key_size, 
            name=prefix+'q_embed'
        )
        self._kv_embed = mlp(
            **self._mlp_config, 
            out_size=kv_size, 
            name=prefix+'kv_embed'
        )

        self._concat = layers.Reshape((seqlen, self._num_heads * self._val_size), name=prefix+'concat')
        self._out =  mlp(
            **self._mlp_config, 
            out_size=out_size, 
            name=prefix+'out'
        )
        if self._drop_rate > 0:
            self._drop = layers.Dropout(self._drop_rate, (None, None, 1), name=prefix+'drop')

        super().build(input_shape)

    def call(self, q, kv, training=False, mask=None):
        N = q.shape[1]
        q = self._q_embed(q)
        kv = self._kv_embed(kv)

        q = tf.reshape(q, (-1, N, self._num_heads, self._key_size))     # [B, N, F] -> [B, N, H, F/H]
        kv = tf.reshape(kv, (-1, N, self._num_heads, self._key_size+self._val_size))    # [B, N, F] -> [B, N, H, F/H]
        q = tf.transpose(q, [0, 2, 1, 3])       # [B, N, H, F/H] -> [B, H, N, F/H]
        kv = tf.transpose(kv, [0, 2, 1, 3])     # [B, N, H, F/H] -> [B, H, N, F/H]

        k, v = tf.split(kv, [self._key_size, self._val_size], -1)
        
        # softmax(QK^T/(d**2))V
        out = compute_dot_product_attention(
            q, k, v, 
            mask=mask, 
            scale_logits=self._scale_logits
        )

        # [B, H, N, V] -> [B, N, H, V]
        out = tf.transpose(out, [0, 2, 1, 3])
        # [B, N, H, V] -> [B, N, H * V]
        x = self._concat(out)
        x = self._out(x)

        if self._drop_rate > 0:
            x = self._drop(x, training=training)

        return x


@nn_registry.register('conv_sa')
@block_registry.register('conv_sa')
class ConvSelfAttention(Module):
    """ Convolutional Self-Attention Module, 
    following SAGAN: https://arxiv.org/abs/1805.08318
    """
    def __init__(
        self,
        key_size=None,
        val_size=None,
        key_ratio=8,
        val_ratio=2,
        scale_logits=False,
        conv='conv2d',
        downsample_ratio=2,
        out_size=None,
        pre_norm=False,
        norm=None,
        norm_kwargs={},
        drop_rate=0,
        use_rezero=True,
        name='conv_sa',
        **kwargs
    ):
        super().__init__(name=name)
        self._key_size = key_size
        self._val_size = val_size
        self._key_ratio = key_ratio
        self._val_ratio = val_ratio
        self._scale_logits = scale_logits
        self._conv = conv
        self._downsample_ratio = downsample_ratio
        self._out_size = out_size
        self._pre_norm = pre_norm
        self._norm = norm
        self._norm_kwargs = norm_kwargs
        self._drop_rate = drop_rate
        self._use_rezero = use_rezero
        kwargs.setdefault('use_bias', False)
        self._kwargs = kwargs
    
    def build(self, input_shape):
        H, W, C = input_shape[1:]
        q_seqlen = kv_seqlen = H * W
        
        key_size, val_size = self._compute_sizes(C)
        self._key_size, self._val_size = key_size, val_size
        
        conv_cls = layer_registry.get(self._conv)
        prefix = f'{self.scope_name}/'

        self._q_conv = conv_cls(key_size, 1, **self._kwargs, name=prefix+'q')
        self._k_conv = conv_cls(key_size, 1, **self._kwargs, name=prefix+'k')
        self._v_conv = conv_cls(val_size, 1, **self._kwargs, name=prefix+'v')

        if self._downsample_ratio > 1:
            self._k_downsample = layers.MaxPool2D(
                self._downsample_ratio, self._downsample_ratio, 
                padding='same', name=prefix+'k_pool')
            self._v_downsample = layers.MaxPool2D(
                self._downsample_ratio, self._downsample_ratio, 
                padding='same', name=prefix+'v_pool')
            kv_seqlen //= self._downsample_ratio**2

        self._q_reshape = layers.Reshape((q_seqlen, key_size), name=prefix+'q_reshape')
        self._k_reshape = layers.Reshape((kv_seqlen, key_size), name=prefix+'k_reshape')
        self._v_reshape = layers.Reshape((kv_seqlen, val_size), name=prefix+'v_reshape')

        self._o_reshape = layers.Reshape((H, W, val_size), name=prefix+'o_reshape')
        self._o_conv = conv_cls(C, 1, **self._kwargs, name=prefix+'o')

        norm_cls = get_norm(self._norm)
        self._norm_layer = norm_cls(**self._norm_kwargs, name=prefix+f'{self._norm}')

        if self._use_rezero:
            self._rezero = tf.Variable(0., trainable=True, dtype=tf.float32, name=prefix+'rezero')
        
        super().build(input_shape)

    def call(self, x, training=False):
        y = call_norm(self._norm, self._norm_layer, x, training) \
            if self._pre_norm else x
        q = self._q_conv(y)
        k = self._k_conv(y)
        v = self._v_conv(y)
        
        if self._downsample_ratio > 1:
            k = self._k_downsample(k)
            v = self._v_downsample(v)

        q = self._q_reshape(q)
        k = self._k_reshape(k)
        v = self._v_reshape(v)

        o = compute_dot_product_attention(
            q, k, v, 
            mask=None, 
            scale_logits=self._scale_logits
        )
        o = self._o_reshape(o)
        o = self._o_conv(o)

        if self._drop_rate > 0:
            y = self._drop(y, training=training)
        if self._use_rezero:
            o = self._rezero * o
        x = o + x
        x = x if self._pre_norm else \
            call_norm(self._norm, self._norm_layer, x, training)

        return x

    def _compute_sizes(self, C):
        if self._key_size is None or self._val_size is None:
            assert self._key_ratio is not None and self._val_ratio is not None
            key_size = C // self._key_ratio
            val_size = C // self._val_ratio
        else:
            key_size = self._key_size
            val_size = self._val_size
        return key_size, val_size


@nn_registry.register('trans_encoder')
class TransformerEncoder(Module):
    def __init__(
        self, 
        n_blocks, 
        att_config,
        mlp_config, 
        name='trans_encoder'
    ):
        super().__init__(name=name)

        att_id = att_config.pop('nn_id')
        if att_id == 'att':
            att_id = 'att2'
        AttCls = block_registry.get(att_id)
        if att_id.startswith('att'):
            att_config.pop('num_heads')
        elif att_id.startswith('mhsa'):
            att_config.setdefault('num_heads', 1)

        self._layers = [
            (AttCls(**att_config, name=name+f'trans_enc_att{i}'), 
            layers.LayerNormalization(name=name+f'trans_enc_ln{i}_1'), 
            mlp(**mlp_config, name=name+f'trans_enc_mlp{i}'), 
            layers.LayerNormalization(name=name+f'trans_enc_ln{i}_2'))
            for i in range(n_blocks)
        ]

    def call(self, x, mask=None):
        for att, ln1, ff, ln2 in self._layers:
            y = att(x, mask=mask)
            x = ln1(x + y)
            y = ff(x)
            x = ln2(x + y)
        return x


@nn_registry.register('trans_decoder')
class TransformerDecoder(Module):
    def __init__(
        self, 
        n_blocks, 
        att_config,
        mlp_config, 
        name='trans_encoder'
    ):
        super().__init__(name=name)

        att_id = att_config.pop('nn_id')
        if att_id.startswith('att'):
            att_config.pop('num_heads')
        elif att_id.startswith('mhsa'):
            att_config.setdefault('num_heads', 1)

        if att_id == 'att':
            att1_id = 'att2'
            att2_id = 'att3'
        else:
            att1_id = 'mhsa'
            att2_id = 'mhsa2'
        AttCls1 = block_registry.get(att1_id)
        AttCls2 = block_registry.get(att2_id)
        self._layers = [
            (AttCls1(**att_config, name=name+f'trans_dec_att{i}_1'), 
            layers.LayerNormalization(name=name+f'trans_dec_ln{i}_1'), 
            AttCls2(**att_config, name=name+f'trans_dec_att{i}_2'), 
            layers.LayerNormalization(name=name+f'trans_dec_ln{i}_2'), 
            mlp(**mlp_config, name=name+f'trans_mlp{i}'), 
            layers.LayerNormalization(name=name+f'trans_dec_ln{i}_3'))
            for i in range(n_blocks)
        ]

    def call(self, x, encoder_x, reverse=False, mask=None):
        for att1, ln1, att2, ln2, ff, ln3 in self._layers:
            y = att1(x, mask=mask)
            x = ln1(x + y)
            if reverse:
                x, encoder_x = encoder_x, x
            y = att2(encoder_x, x, mask=mask)
            x = ln2(x + y)
            y = ff(x)
            x = ln3(x + y)
        return x


if __name__ == "__main__":
    shape = (3, 4, 4, 2)
    # x = layers.Input(shape)
    tf.random.set_seed(0)
    x = tf.random.normal(shape)
    sa = ConvSelfAttention()
    y = sa(x)
    # import time
    # start = time.time()
    # for _ in range(100):
    #     x = tf.random.normal(shape)
    #     y = sa(x)
    # print(time.time() - start)
    print(sa.variables)
    # model = tf.keras.Model(x, y)
    # model.summary(200)
