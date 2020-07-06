import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from core.module import Module
from core.decorator import config
from nn.func import mlp, cnn
        

class FractionProposalNetwork(Module):
    @config
    def __init__(self, name='fqn'):
        super().__init__(name=name)
        kwargs = {}
        if hasattr(self, '_kernel_initializer'):
            kwargs['kernel_initializer'] = self._kernel_initializer
        self._layers = mlp(
            out_size=self.N,
            name='fpn',
            **kwargs)
    
    def __call__(self, x):
        x = self._layers(x)

        log_probs = tf.nn.log_softmax(x, axis=-1)
        probs = tf.exp(log_probs)
        entropy = -probs * log_probs

        tau_0 = tf.zeros([*probs.shape[:-1], 1])
        tau_rest = tf.math.cumsum(probs, axis=-1)

        tau = tf.concat([tau_0, tau_rest], axis=-1)          # [B, N+1]
        tau_hat = (tau[..., :-1] + tau[..., 1:]) / 2.   # [B, N]

        tf.debugging.assert_shapes([
            [tau_0, (None, 1)],
            [probs, (None, self.N)],
            [tau, (None, self.N+1)],
            [tau_hat, (None, self.N)]
        ])

        return tau, tau_hat, entropy


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

    def __call__(self, x, deterministic=False, epsilon=0):
        x = np.array(x)
        if not deterministic and np.random.uniform() < epsilon:
            size = x.shape[0] if len(x.shape) % 2 == 0 else None
            return np.random.randint(self._action_dim, size=size)
        if len(x.shape) % 2 != 0:
            x = tf.expand_dims(x, 0)
        
        action = self.action(x)
        action = np.squeeze(action.numpy())

        return action

    def action(self, x, tau_hat, tau_range=None):
        _, q = self.value(x, tau_hat, tau_range)
        return tf.argmax(q, axis=-1, output_type=tf.int32)

    def value(self, x, tau_hat, tau_range=None, action=None):
        tf.debugging.assert_shapes([
            [x, (None, self._cnn.out_size)],
        ])
        x = tf.expand_dims(x, 1)   # [B, 1, cnn.out_size]
        qt_embed = self.qt_embed(tau_hat)   # [B, N, cnn.out_size]
        x = x * qt_embed    # [B, N, cnn.out_size]
        qtv = self.qtv(x, tau_hat, action=action)
        if tau_range is None:
            return qtv
        else:
            q = self.q(qtv, tau_range, action=action)
            return qtv, q

    def cnn(self, x):
        # psi network
        if self._cnn:
            x = self._cnn(x)
        return x
    
    def qt_embed(self, tau_hat):
        # phi network
        tau_hat = tf.expand_dims(tau_hat, -1)       # [B, N, 1]
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        degree = tf.cast(tf.range(self._qt_embed_size), tf.float32) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        qt_embed = self._phi(qt_embed)              # [B, N, cnn.out_size]
        
        return qt_embed

    def qtv(self, x, tau, action=None):
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
            qtv = tf.reduce_sum(qtv * action, -1, keepdims=True)        # [B, N, 1], relying on broadcasting
            
        return qtv

    def q(self, qtv, tau_range, action=None):
        diff = tau_range[..., 1:] - tau_range[..., :-1]
        diff = tf.expand_dims(diff, axis=-1)
        q = tf.reduce_sum(diff * qtv, axis=1)                       # [B, A]
        if action is not None:
            if len(action.shape) < len(qtv.shape) - 1:
                action = tf.one_hot(action, self._action_dim, dtype=q.dtype)
            q = tf.reduce_sum(q * action, -1)                           # [B]
            tf.debugging.assert_shapes([
                [action, (None, self._action_dim)],
                [q, (None)],
            ])
        return q


def create_model(config, env, **kwargs):
    action_dim = env.action_dim
    fpn = FractionProposalNetwork(config['fpn'], name='fpn')
    q = Q(config['iqn'], action_dim, name='iqn')
    target_q = Q(config['iqn'], action_dim, name='target_iqn')
    return dict(
        fpn=fpn,
        q=q,
        target_q=target_q,
    )
