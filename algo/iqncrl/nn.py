import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

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
        self._cnn = cnn(self._cnn, out_size=self._cnn_out_size, **kwargs)

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

    def value(self, x, n_qt=None, action=None, return_obs_embed=False):
        if n_qt is None:
            n_qt = self.K
        batch_size = x.shape[0]
        z = self.cnn(x)
        x = tf.expand_dims(z, 1)    # [B, 1, cnn.out_size]
        tau_hat, qt_embed = self.quantile(n_qt, batch_size, x.shape[-1])
        x = x * qt_embed            # [B, N, cnn.out_size]
        qtv = self.qtv(x, action=action)
        q = self.q(qtv)

        if return_obs_embed:
            return z, tau_hat, qtv, q
        else:
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
        degree = tf.cast(tf.range(self._tau_embed_size), tf.float32) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        tf.debugging.assert_shapes([
            [tau_hat, (batch_size, n_qt, 1)],
            [qt_embed, (batch_size, n_qt, self._tau_embed_size)],
        ])
        if not hasattr(self, '_phi'):
            self._phi = mlp(
                [cnn_out_size], 
                activation=self._phi_activation,
                name='phi',
                **self._kwargs)
        qt_embed = self._phi(qt_embed)              # [B, N, cnn.out_size]
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

class CRL(Module):
    @config
    def __init__(self,):
        self._crl_mlp = mlp(
            self._crl_units,
            out_size=self._crl_out_size,
            activation=self._activation,
            out_dtype='float32',
            name='crl',
        )
        self._crl_w = tf.Variable(tf.random.uniform((self._crl_out_size, self._crl_out_size)))

    def __call__(self, x):
        z = self._crl_mlp(x)
        return z

    def logits(self, x_anchor, x_pos):
        x_pos = tf.stop_gradient(x_pos)
        Wx = tf.matmul(self._crl_w, tf.transpose(x_pos))
        logits = tf.matmul(x_anchor, Wx)
        logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        return logits


class IQNCRL(Ensemble):
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

        if epsilon > 0:
            rand_act = tf.random.uniform(
                action.shape, 0, self.q.action_dim, dtype=tf.int32)
            action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)
        action = tf.squeeze(action)
        qtv = tf.squeeze(qtv)

        return action, {'qtv': qtv}

def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    return dict(
        q=Q(config['q'], action_dim, 'q'),
        target_q=Q(config['q'], action_dim, 'target_q'),
        crl=CRL(config['crl'])
    )

def create_model(config, env, **kwargs):
    return IQNCRL(config, env, **kwargs)