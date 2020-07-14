import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp, cnn
        

class CNN(Module):
    @config
    def __init__(self, name='cnn'):
        kwargs = dict(
            out_size=self._cnn_out_size,
            kernel_initializer=self._kernel_initializer,
        )
        self._cnn = cnn(self._cnn, **kwargs)

    def __call__(self, x):
        x = self._cnn(x)
        return x


class Actor(Module):
    @config
    def __init__(self, action_dim, name='actor'):
        super().__init__(name=name)
        
        self._layers = mlp(self._units_list, 
                            out_size=action_dim,
                            activation=self._activation)

    def __call__(self, x, deterministic=False, epsilon=0):
        x = self._layers(x)

        dist = tfd.Categorical(logits=x)
        action = dist.mode() if deterministic else dist.sample()
        if epsilon > 0:
            rand_act = tfd.Categorical(tf.zeros_like(dist.logits)).sample()
            action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)

        return action

    def train_step(self, x):
        x = self._layers(x)
        probs = tf.nn.softmax(x)
        logps = tf.math.log(tf.maximum(probs, 1e-8))    # bound logps to avoid numerical instability
        return probs, logps


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

    def __call__(self, x, n_qt=None, action=None, return_q=False):
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
        degree = tf.cast(tf.range(self._qt_embed_size), tf.float32) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        tf.debugging.assert_shapes([
            [tau_hat, (batch_size, n_qt, 1)],
            [qt_embed, (batch_size, n_qt, self._qt_embed_size)],
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


class Temperature(Module):
    @config
    def __init__(self, name='temperature'):
        super().__init__(name=name)

        if self._temp_type == 'state-action':
            self._layer = layers.Dense(1)
        elif self._temp_type == 'variable':
            self._log_temp = tf.Variable(
                np.log(self._value), dtype=tf.float32, name='log_temp')
        else:
            raise NotImplementedError(f'Error temp type: {self._temp_type}')
    
    def __call__(self, x=None, a=None):
        if self._temp_type == 'state-action':
            x = tf.concat([x, a], axis=-1)
            x = self._layer(x)
            log_temp = -tf.nn.softplus(x)
            log_temp = tf.squeeze(log_temp)
        else:
            log_temp = self._log_temp
        temp = tf.exp(log_temp)
    
        return log_temp, temp


class SACIQN(Ensemble):
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

        x = self.cnn(x)
        action = self.actor(x, deterministic=deterministic, epsilon=epsilon)
        _, qtv = self.q(x, action=action)
        action = tf.squeeze(action)
        qtv = tf.squeeze(qtv)

        return action, {'qtv': qtv}


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    temperature_config = config['temperature']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    return dict(
        cnn=CNN(config['cnn'], name='cnn'),
        actor=Actor(config['actor'], action_dim, name='actor'),
        q=Q(config['q'], action_dim, name='q'),
        target_q=Q(config['q'], action_dim, name='target_q'),
        temperature=temperature,
    )

def create_model(config, env, **kwargs):
    return SACIQN(config, env, **kwargs)
