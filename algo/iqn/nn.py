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
        kwargs = {}
        if hasattr(self, '_kernel_initializer'):
            kwargs['kernel_initializer'] = self._kernel_initializer
        self._kwargs = kwargs
        self._cnn = cnn(self._cnn_name, out_size=self._out_size, **kwargs)

        # we do not define the phi net here to make it consistent with the CNN output size
        if self._duel:
            self._v_head = mlp(
                self._units_list, 
                out_size=1, 
                activation=self._activation, 
                out_dtype='float32',
                name='v',
                **kwargs)
        self._a_head = mlp(
            self._units_list, 
            out_size=action_dim, 
            activation=self._activation, 
            out_dtype='float32',
            name='a' if self._duel else 'q',
            **kwargs)

    @property
    def action_dim(self):
        return self._action_dim

    def action(self, x, n_qt=None):
        _, _, q = self.value(x, n_qt)
        return tf.argmax(q, axis=-1, output_type=tf.int32)
    
    def value(self, x, n_qt=None, action=None):
        if n_qt is None:
            n_qt = self.K
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = tf.expand_dims(x, 1)    # [B, 1, cnn.out_size]
        tau_hat, qt_embed = self.quantile(n_qt, batch_size, x.shape[-1])
        x = x * qt_embed            # [B, N, cnn.out_size]
        qtv = self.qtv(x, action=action)
        q = self.q(qtv)
        
        return tau_hat, qtv, q

    def cnn(self, x):
        # psi network
        if self._cnn:
            x = self._cnn(x)
        return x
    
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
        qt_embed = self.mlp(qt_embed, [cnn_out_size], 
                activation=self._phi_activation,
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
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        _, qtv, q = self.q.value(x)
        action = tf.argmax(q, axis=-1, output_type=tf.int32)
        qtv = tf.math.reduce_max(qtv, -1)

        action = epsilon_greedy(action, epsilon,
            is_action_discrete=True, action_dim=self.q.action_dim)
        action = tf.squeeze(action)
        qtv = tf.squeeze(qtv)

        return action, {'qtv': qtv}

def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    return dict(
        q=Q(config, action_dim, 'q'),
        target_q=Q(config, action_dim, 'target_q'),
    )

def create_model(config, env, **kwargs):
    return IQN(config, env, **kwargs)
