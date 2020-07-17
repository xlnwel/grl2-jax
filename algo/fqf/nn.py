import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp, cnn
        

class FractionProposalNetwork(Module):
    @config
    def __init__(self, name='fqn'):
        super().__init__(name=name)
        kwargs = {}
        if hasattr(self, '_kernel_initializer'):
            kwargs['kernel_initializer'] = self._kernel_initializer
            kwargs['gain'] = .01
        self._layers = mlp(
            out_size=self.N,
            name='fpn',
            **kwargs)
    
    def __call__(self, x):
        x = self._layers(x)

        log_probs = tf.nn.log_softmax(x, axis=-1)
        probs = tf.exp(log_probs)
        entropy = -probs * log_probs

        tau_0 = tf.zeros([*probs.shape[:-1], 1], dtype=probs.dtype)
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

    def action(self, x, tau_hat, tau_range=None):
        _, q = self.value(x, tau_hat, tau_range)
        return tf.argmax(q, axis=-1, output_type=tf.int32)

    def value(self, x, tau_hat, tau_range=None, action=None):
        x = tf.expand_dims(x, 1)    # [B, 1, cnn.out_size]
        cnn_out_size = x.shape[-1]
        qt_embed = self.qt_embed(tau_hat, cnn_out_size)   # [B, N, cnn.out_size]
        x = x * qt_embed            # [B, N, cnn.out_size]
        qtv = self.qtv(x, action=action)
        if tau_range is None:
            return qtv
        else:
            q = self.q(qtv, tau_range)
            return qtv, q

    def cnn(self, x):
        # psi network
        if self._cnn:
            x = self._cnn(x)
        return x
    
    def qt_embed(self, tau_hat, cnn_out_size):
        # phi network
        tau_hat = tf.expand_dims(tau_hat, axis=-1)       # [B, N, 1]
        pi = tf.convert_to_tensor(np.pi, dtype=tau_hat.dtype)
        degree = tf.cast(tf.range(self._qt_embed_size), tau_hat.dtype) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        
        if not hasattr(self, '_phi'):
            self._phi = mlp(
                [cnn_out_size], 
                activation=self._phi_activation,
                name='phi',
                **self._kwargs)
        qt_embed = self._phi(qt_embed)              # [B, N, cnn.out_size]
        
        return qt_embed

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
            qtv = tf.reduce_sum(qtv * action, axis=-1)        # [B, N]
            
        return qtv

    def q(self, qtv, tau_range):
        diff = tau_range[..., 1:] - tau_range[..., :-1]
        if len(qtv.shape) > len(diff.shape):
            diff = tf.expand_dims(diff, axis=-1)        # expand diff if qtv includes the action dimension
        q = tf.reduce_sum(diff * qtv, axis=1)           # [B, A] / [B]
        
        return q


class FQF(Ensemble):
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

        x = self.q.cnn(x)
        tau, tau_hat, _ = self.fpn(x)
        qtv, q = self.q.value(x, tau_hat, tau_range=tau)
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

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.q.cnn(x)
        tau, tau_hat, _ = self.fpn(x)
        qtv, q = self.q.value(x, tau_hat, tau_range=tau)
        qtv = tf.squeeze(qtv)
        q = tf.squeeze(q)

        return qtv, q


def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    fpn = FractionProposalNetwork(config['fpn'], name='fpn')
    q = Q(config['iqn'], action_dim, name='iqn')
    target_q = Q(config['iqn'], action_dim, name='target_iqn')
    return dict(
        fpn=fpn,
        q=q,
        target_q=target_q,
    )

def create_model(config, env, **kwargs):
    return FQF(config, env, **kwargs)
