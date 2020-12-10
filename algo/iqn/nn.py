import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import epsilon_greedy
from nn.func import Encoder, mlp
        

class Quantile(Module):
    @config
    def __init__(self, name='phi'):
        super().__init__(name=name)

    def call(self, x, n_qt=None):
        batch_size, cnn_out_size = x.shape
        # phi network
        n_qt = n_qt or self.K
        tau_hat = tf.random.uniform([batch_size, n_qt, 1], 
            minval=0, maxval=1, dtype=tf.float32)   # [B, N, 1]
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        degree = tf.cast(tf.range(1, self._tau_embed_size+1), tau_hat.dtype) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            out_dtype='float32',
        )
        qt_embed = self.mlp(
            qt_embed, 
            [cnn_out_size], 
            name=self.name,
            **kwargs)                  # [B, N, cnn.out_size]
        tf.debugging.assert_shapes([
            [qt_embed, (batch_size, n_qt, cnn_out_size)],
        ])
        return tau_hat, qt_embed


class Value(Module):
    @config
    def __init__(self, action_dim, name='value'):
        super().__init__(name=name)

        self._action_dim = action_dim

        """ Network definition """
        kwargs = dict(
            layer_type=getattr(self, '_layer_type', 'dense'),
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            out_dtype='float32',
        )
        self._kwargs = kwargs

        # we do not define the phi net here to make it consistent with the CNN output size
        self._layers = mlp(
            self._units_list,
            out_size=action_dim, 
            name=name,
            **kwargs)

    @property
    def action_dim(self):
        return self._action_dim

    def action(self, x, qt_embed=None, tau_range=None, return_stats=False):
        qtv, q = self.call(x, qt_embed, tau_range=tau_range, return_value=True)
        action = tf.argmax(q, axis=-1, output_type=tf.int32)
        if return_stats:
            return action, qtv, q
        else:
            return action
    
    def call(self, x, qt_embed, action=None, tau_range=None, return_value=False):
        if x.shape.ndims < qt_embed.shape.ndims:
            x = tf.expand_dims(x, axis=1)
        assert x.shape.ndims == qt_embed.shape.ndims, (x.shape, qt_embed.shape)
        x = x * qt_embed            # [B, N, cnn.out_size]
        qtv = self.qtv(x, action=action)
        if tau_range is not None or return_value:
            v = self.value(qtv, tau_range=tau_range)
            return qtv, v
        else:
            return qtv

    def qtv(self, x, action=None):
        qtv = self._layers(x)   # [B, N, A]

        if action is not None:
            assert self.action_dim != 1, f"action is not None when action_dim = {self.action_dim}"
            action = tf.expand_dims(action, axis=1)
            if len(action.shape) < len(qtv.shape):
                action = tf.one_hot(action, self._action_dim, dtype=qtv.dtype)
            qtv = tf.reduce_sum(qtv * action, axis=-1)       # [B, N]
            
        return qtv

    def value(self, qtv, tau_range=None):
        if tau_range is None:
            v = tf.reduce_mean(qtv, axis=1)     # [B, A] / [B]
        else:
            diff = tau_range[..., 1:] - tau_range[..., :-1]
            if len(qtv.shape) > len(diff.shape):
                diff = tf.expand_dims(diff, axis=-1)        # expand diff if qtv includes the action dimension
            v = tf.reduce_sum(diff * qtv, axis=1)

        return v


class IQN(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0, return_stats=False):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        _, qt_embed = self.quantile(x)
        action = self.q.action(x, qt_embed, return_stats=return_stats)
        action = epsilon_greedy(action, epsilon,
            is_action_discrete=True, action_dim=self.q.action_dim)
        action = tf.squeeze(action)
        terms = {}
        if return_stats:
            action, _, q = action
            q = tf.squeeze(q)
            terms = {'q': q}

        return action, terms


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    encoder_config = config['encoder']
    quantile_config = config['quantile']
    q_config = config['q']
    
    return dict(
        encoder=Encoder(encoder_config, name='encoder'),
        quantile=Quantile(quantile_config, name='phi'),
        q=Value(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_quantile=Quantile(quantile_config, name='target_phi'),
        target_q=Value(q_config, action_dim, name='target_q'),
    )

def create_model(config, env, **kwargs):
    return IQN(config, env=env, **kwargs)
