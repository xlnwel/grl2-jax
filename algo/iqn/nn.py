import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import epsilon_greedy
from nn.func import mlp, cnn
        

class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)

        self._action_dim = action_dim

        """ Network definition """
        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            out_dtype='float32',
        )
        self._kwargs = kwargs

        # we do not define the phi net here to make it consistent with the CNN output size
        if self._duel:
            self._v_head = mlp(
                self._units_list, 
                out_size=1, 
                layer_type=self._layer_type,
                norm=self._norm,
                name='v',
                **kwargs)
        self._a_head = mlp(
            self._units_list, 
            out_size=action_dim, 
            layer_type=self._layer_type,
            norm=self._norm,
            name='a' if self._duel else 'q',
            **kwargs)

    @property
    def action_dim(self):
        return self._action_dim

    def action(self, x, n_qt=None, return_stats=False):
        _, qtv, q = self(x, n_qt)
        action = tf.argmax(q, axis=-1, output_type=tf.int32)
        if return_stats:
            return action, qtv, q
        else:
            return action
    
    def call(self, x, n_qt=None, action=None, return_q=False):
        if n_qt is None:
            n_qt = self.K
        batch_size = x.shape[0]
        x = tf.expand_dims(x, 1)    # [B, 1, cnn.out_size]
        tau_hat, qt_embed = self.quantile(n_qt, batch_size, x.shape[-1])
        x = x * qt_embed            # [B, N, cnn.out_size]
        qtv = self.qtv(x, action=action)
        if return_q:
            q = self.q(qtv)
            return tau_hat, qtv, q
        else:
            return tau_hat, qtv
    
    def quantile(self, n_qt, batch_size, cnn_out_size):
        # phi network
        tau_hat = tf.random.uniform([batch_size, n_qt, 1], 
            minval=0, maxval=1, dtype=tf.float32)   # [B, N, 1]
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        # start from 1 since degree of 0 is meaningless
        degree = tf.cast(tf.range(1, self._tau_embed_size+1), tau_hat.dtype) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        tf.debugging.assert_shapes([
            [tau_hat, (batch_size, n_qt, 1)],
            [qt_embed, (batch_size, n_qt, self._tau_embed_size)],
        ])
        qt_embed = self.mlp(
            qt_embed, 
            [cnn_out_size], 
            name='phi',
            **self._kwargs)                  # [B, N, cnn.out_size]
        tf.debugging.assert_shapes([
            [qt_embed, (batch_size, n_qt, cnn_out_size)],
        ])
        return tau_hat, qt_embed

    def qtv(self, x, action=None):
        if self._duel:
            v_qtv = self._v_head(x) # [B, N, 1]
            a_qtv = self._a_head(x) # [B, N, A]
            qtv = v_qtv + a_qtv - tf.reduce_mean(a_qtv, axis=-1, keepdims=True)
        else:
            qtv = self._a_head(x)   # [B, N, A]

        if action is not None:
            action = tf.expand_dims(action, axis=1)
            if len(action.shape) < len(qtv.shape):
                action = tf.one_hot(action, self._action_dim, dtype=qtv.dtype)
            qtv = tf.reduce_sum(qtv * action, axis=-1)       # [B, N]

        return qtv

    def q(self, qtv):
        q = tf.reduce_mean(qtv, axis=1)     # [B, A] / [B]
        
        return q


class IQN(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0, return_stats=False):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        action = self.q.action(x, return_stats=return_stats)
        terms = {}
        if return_stats:
            action, qtv, q = action
            qtv = tf.math.reduce_max(qtv, -1)
            qtv = tf.squeeze(qtv)
            terms = {'qtv': qtv}

        action = epsilon_greedy(action, epsilon,
            is_action_discrete=True, action_dim=self.q.action_dim)
        action = tf.squeeze(action)

        return action, terms


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    encoder_config = config['encoder']
    q_config = config['q']
    
    return dict(
        encoder=cnn(encoder_config, name='encoder'),
        target_encoder=cnn(encoder_config, name='target_encoder'),
        q=Q(q_config, action_dim, name='q'),
        target_q=Q(q_config, action_dim, name='target_q'),
    )

def create_model(config, env, **kwargs):
    return IQN(config, env, **kwargs)
