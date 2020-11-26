import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from algo.sacd.nn import Encoder, Actor, Temperature


class Quantile(Module):
    @config
    def __init__(self, name='phi'):
        super().__init__(name=name)

        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            out_dtype='float32',
        )
        self._kwargs = kwargs

    def call(self, x, n_qt=None):
        batch_size, cnn_out_size = x.shape
        # phi network
        n_qt = n_qt or self.K
        tau_hat = tf.random.uniform([batch_size, n_qt, 1], 
            minval=0, maxval=1, dtype=tf.float32)   # [B, N, 1]
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        degree = tf.cast(tf.range(1, self._tau_embed_size+1), tau_hat.dtype) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        qt_embed = self.mlp(
            qt_embed, 
            [cnn_out_size], 
            name=self.name,
            **self._kwargs)                  # [B, N, cnn.out_size]
        tf.debugging.assert_shapes([
            [qt_embed, (batch_size, n_qt, cnn_out_size)],
        ])
        return tau_hat, qt_embed
    
class Value(Module):
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
        self._layers = mlp(
            self._units_list,
            out_size=action_dim, 
            layer_type=self._layer_type,
            norm=self._norm,
            name=name,
            **kwargs)

    @property
    def action_dim(self):
        return self._action_dim
    
    def call(self, x, qt_embed, action=None, return_value=False):
        batch_size = x.shape[0]
        assert x.shape.ndims == qt_embed.shape.ndims, (x.shape, qt_embed.shape)
        x = x * qt_embed            # [B, N, cnn.out_size]
        qtv = self.qtv(x, action=action)
        if return_value:
            v = self.value(qtv)
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

    def value(self, qtv):
        v = tf.reduce_mean(qtv, axis=1)     # [B, A] / [B]

        return v


class SACIQN(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        self._reward_entropy = config.pop('reward_entropy', False)
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
        action = self.actor(x, deterministic=deterministic, epsilon=epsilon)
        terms = {}
        if return_eval_stats:
            action, terms = action
            _, qt_embed = self.quantile(x)
            _, qtv, q = self.q(x, qt_embed, return_q=True)
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
            _, qt_embed = self.quantile(x)
            x_ext = tf.expand_dims(x, axis=1)
            # _, v = self.v(x_ext, qt_embed, return_value=True)
            _, v = self.q(x_ext, qt_embed, action=action, return_value=True)
            if self._reward_entropy:
                logp = tfd.Categorical(self.actor.logits).log_prob(action)
                if self.temperature.type == 'schedule':
                    _, temp = self.temperature(self._train_step)
                elif self.temperature.type == 'state-action':
                    _, temp = self.temperature(x, action)
                else:
                    _, temp = self.temperature()
                logp = temp * logp
                terms['logp'] = logp
            v = tf.squeeze(v)
            terms['v'] = v
        action = tf.squeeze(action)
        return action, terms


def create_components(config, env, **kwargs):
    assert env.is_action_discrete, env.name
    action_dim = env.action_dim
    encoder_config = config['encoder']
    actor_config = config['actor']
    v_config = config['v']
    q_config = config['q']
    temperature_config = config['temperature']
    quantile_config = config['quantile']

    models = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        quantile=Quantile(quantile_config, name='phi'),
        actor=Actor(actor_config, action_dim),
        v=Value(v_config, 1, name='v'),
        q=Value(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_actor=Actor(actor_config, action_dim, name='target_actor'),
        target_v=Value(v_config, 1, name='target_v'),
        target_q=Value(q_config, action_dim, name='target_q'),
        temperature=Temperature(temperature_config),
    )

    return models

def create_model(config, env, **kwargs):
    return SACIQN(config=config, env=env, **kwargs)
