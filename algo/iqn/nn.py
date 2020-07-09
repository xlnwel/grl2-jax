import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from core.module import Module, Ensemble
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
        self._kwargs = kwargs
        self._cnn = cnn(self._cnn, out_size=None, **kwargs)

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

    @property
    def action_dim(self):
        return self._action_dim

    @tf.function
    def action(self, x, n_qt=None):
        _, _, q = self.value(x, n_qt)
        return tf.argmax(q, axis=-1, output_type=tf.int32)
    
    @tf.function
    def value(self, x, n_qt=None, action=None):
        if n_qt is None:
            n_qt = self.K
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = tf.tile(x, [n_qt, 1])   # [N*B, cnn.out_size]
        tau_hat, qt_embed = self.quantile(n_qt, batch_size, x.shape[-1])
        x = x * qt_embed    # [N*B, cnn.out_size]
        qtv = self.qtv(x, n_qt, batch_size, action=action)
        q = self.q(qtv, action=action)
        
        return tau_hat, qtv, q

    def cnn(self, x):
        # psi network
        if self._cnn:
            x = self._cnn(x)
        return x
    
    def quantile(self, n_qt, batch_size, cnn_out_size):
        # phi network
        tau_hat = tf.random.uniform([n_qt * batch_size, 1], 
            minval=0, maxval=1, dtype=tf.float32)
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        degree = tf.cast(tf.range(self._qt_embed_size), tf.float32) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [N*B, E]
        if not hasattr(self, '_phi'):
            self._phi = mlp(
                [cnn_out_size], 
                activation=self._phi_activation,
                name='phi',
                **self._kwargs)
        qt_embed = self._phi(qt_embed)              # [N*B, cnn.out_size]
        
        return tau_hat, qt_embed

    def qtv(self, x, n_qt, batch_size, action=None):
        if self._duel:
            v_qtv = self._v_head(x) # [N*B, 1]
            a_qtv = self._a_head(x) # [N*B, A]
            qtv = v_qtv + a_qtv - tf.reduce_mean(a_qtv, axis=-1, keepdims=True)
        else:
            qtv = self._a_head(x)   # [N*B, A]
        qtv = tf.reshape(qtv, (n_qt, batch_size, self._action_dim))     # [N, B, A]
        qtv = tf.transpose(qtv, [1, 0, 2])                              # [B, N, A]
        
        if action is not None:
            action = tf.expand_dims(action, axis=1)
            if len(action.shape) < len(qtv.shape):
                action = tf.one_hot(action, self._action_dim, dtype=qtv.dtype)
            qtv = tf.reduce_sum(qtv * action, -1, keepdims=True)        # [B, N, 1], relying on broadcasting
            
        return qtv

    def q(self, qtv, action=None):
        q = tf.reduce_mean(qtv, axis=1)     # [B, A]

        if action is not None:
            if len(action.shape) < len(q.shape):
                action = tf.one_hot(action, self._action_dim, dtype=q.dtype)
            q = tf.reduce_sum(q * action, axis=-1)  #[B]
            tf.debugging.assert_shapes([
                [action, (None, self._action_dim)],
                [q, (None)],
            ])
        
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

        tau_hat, qtv, q = self.q.value(x)
        action = tf.argmax(q, axis=-1, output_type=tf.int32)
        if not deterministic and epsilon > 0:
            rand_act = tf.random.uniform(
                action.shape, 0, self.q.action_dim, dtype=tf.int32)
            action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)
        action = tf.squeeze(action)

        return action, {'qtv': qtv}

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        tau_hat, qtv, q = self.q.value(x)

        return qtv, q

def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    q = Q(config, action_dim, 'q')
    target_q = Q(config, action_dim, 'target_q')
    return dict(
        q=q,
        target_q=target_q,
    )

def create_model(config, env, **kwargs):
    return IQN(config, env, **kwargs)
