import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from core.module import Module
from core.decorator import config
from nn.func import mlp, cnn
        

class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)
        self._dtype = global_policy().compute_dtype

        self._action_dim = action_dim

        """ Network definition """
        kwargs = {}
        if hasattr(self, '_kernel_initializer'):
            kwargs['kernel_initializer'] = self._kernel_initializer
        self._cnn = cnn(self._cnn, **kwargs)

        self._phi = mlp(
            [self._cnn.out_size], 
            activation=self._phi_activation,
            name='phi',
            **kwargs)
        if self._duel:
            self._v_head = mlp(
                self._head_units, 
                out_size=1, 
                activation=self._activation, 
                out_dtype='float32',
                name='v',
                **kwargs)
        self._a_head = mlp(
            self._head_units, 
            out_size=action_dim, 
            activation=self._activation, 
            out_dtype='float32',
            name='a' if self._duel else 'q',
            **kwargs)

    def __call__(self, x, n_qt, deterministic=False, epsilon=0):
        x = np.array(x)
        if not deterministic and np.random.uniform() < epsilon:
            size = x.shape[0] if len(x.shape) % 2 == 0 else None
            return np.random.randint(self._action_dim, size=size)
        if len(x.shape) % 2 != 0:
            x = tf.expand_dims(x, 0)
        
        action = self.action(x, n_qt)
        action = np.squeeze(action.numpy())

        return action

    @tf.function
    def action(self, x, n_qt):
        _, _, q = self.value(x, n_qt)
        return tf.argmax(q, axis=-1, output_type=tf.int32)
    
    @tf.function
    def value(self, x, n_qt, action=None):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = tf.tile(x, [n_qt, 1])   # [N*B, cnn.out_size]
        tau_hat, qt_embed = self.quantile(n_qt, batch_size)
        x = x * qt_embed    # [N*B, cnn.out_size]
        qtv, q = self.mlp(x, n_qt, batch_size, action=action)
        
        return tau_hat, qtv, q

    def cnn(self, x):
        # psi network
        if self._cnn:
            x = self._cnn(x)
        return x
    
    def quantile(self, n_qt, batch_size):
        tau_hat = tf.random.uniform([n_qt * batch_size, 1], 
            minval=0, maxval=1, dtype=tf.float32)
        qt_tiled = tf.tile(tau_hat, [1, self._qt_embed_size])  # [N*B, E]

        # phi network
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        degree = tf.cast(tf.range(self._qt_embed_size), tf.float32) * pi * qt_tiled
        qt_embed = tf.math.cos(degree)              # [N*B, E]
        qt_embed = self._phi(qt_embed)              # [N*B, cnn.out_size]
        
        return tau_hat, qt_embed

    def mlp(self, x, n_qt, batch_size, action=None):
        if self._duel:
            v_qtv = self._v_head(x)
            a_qtv = self._a_head(x)
            qtv = v_qtv + a_qtv - tf.reduce_mean(a_qtv, axis=-1, keepdims=True)
        else:
            qtv = self._a_head(x)
        qtv = tf.reshape(qtv, (n_qt, batch_size, self._action_dim))     # [N, B, A]
        q = tf.reduce_mean(qtv, axis=0)                                 # [B, A]
        
        if action is not None:
            if len(action.shape) < len(q.shape):
                action = tf.one_hot(action, self._action_dim, dtype=q.dtype)
            action_qtv = tf.tile(action, [n_qt, 1])
            action_qtv = tf.reshape(action_qtv, [n_qt, batch_size, self._action_dim])
            qtv = tf.reduce_sum(qtv * action, -1, keepdims=True)        # [N, B, 1]
            q = tf.reduce_sum(q * action, -1)                           # [B]
        return qtv, q


def create_model(config, env, **kwargs):
    action_dim = env.action_dim
    q = Q(config, action_dim, 'q')
    target_q = Q(config, action_dim, 'target_q')
    return dict(
        q=q,
        target_q=target_q,
    )
