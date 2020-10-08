import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from algo.sacd.nn import Encoder, Actor, Temperature


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
        degree = tf.cast(tf.range(self._tau_embed_size), tf.float32) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
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


class SACIQN(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0, return_stats=False, return_eval_stats=False, **kwargs):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        action = self.actor(x, deterministic=deterministic, epsilon=epsilon, return_stats=return_eval_stats)
        terms = {}
        if return_eval_stats:
            action, terms = action
            _, qtv, q = self.q(x, return_q=True)
            qtv = tf.transpose(qtv, [0, 2, 1])
            idx = tf.stack([tf.range(action.shape[0]), action], -1)
            qtv_max = tf.reduce_max(qtv, 1)
            q_max = tf.reduce_max(q, 1)
            action_best_q = tf.argmax(q, 1)
            qtv = tf.gather_nd(qtv, idx)
            q = tf.gather_nd(q, idx)
            action = tf.squeeze(action)
            action_best_q = tf.squeeze(action_best_q)
            qtv_max = tf.squeeze(qtv_max)
            qtv = tf.squeeze(qtv)
            q_max = tf.squeeze(q_max)
            q = tf.squeeze(q)
            terms = {
                'action': action,
                'action_best_q': action_best_q,
                'qtv_max': qtv_max,
                'qtv': qtv,
                'q_max': q_max,
                'q': q,
            }
        elif return_stats:
            _, qtv = self.q(x, action=action)
            qtv = tf.squeeze(qtv)
            terms['qtv'] = qtv
        action = tf.squeeze(action)

        return action, terms


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    encoder_config = config['encoder']
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    models = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        actor=Actor(actor_config, action_dim),
        target_actor=Actor(actor_config, action_dim),
        q=Q(q_config, action_dim, name='q'),
        target_q=Q(q_config, action_dim, name='target_q'),
        temperature=temperature,
    )
    if config['twin_q']:
        models['q2'] = Q(q_config, action_dim, name='q2')
        models['target_q2'] = Q(q_config, action_dim, name='target_q2')

    return models

def create_model(config, env, **kwargs):
    return SACIQN(config=config, env=env, **kwargs)
