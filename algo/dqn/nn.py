import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import epsilon_greedy
from nn.func import Encoder, mlp
from nn.layers import Noisy


class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)

        self.action_dim = action_dim

        """ Network definition """
        kwargs = {}
        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
        )
        
        if self._duel:
            self._v_head = mlp(
                self._units_list, 
                layer_type=self._layer_type,
                out_size=1, 
                name='v',
                **kwargs)
        self._a_head = mlp(
            self._units_list, 
            layer_type=self._layer_type,
            out_size=action_dim, 
            name='a' if self._duel else 'q',
            **kwargs)

    def action(self, x, noisy=True, reset=True):
        q = self.call(x, noisy=noisy, reset=reset)
        return tf.argmax(q, axis=-1, output_type=tf.int32)
    
    def call(self, x, action=None, noisy=True, reset=True):
        kwargs = dict(noisy=noisy, reset=reset) if self._layer_type == 'noisy' else {}

        if self._duel:
            v = self._v_head(x, **kwargs)
            a = self._a_head(x, **kwargs)
            q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        else:
            q = self._a_head(x, **kwargs)

        if action is not None:
            if len(action.shape) < len(q.shape):
                action = tf.one_hot(action, self.action_dim, dtype=q.dtype)
            assert q.shape[-1] == action.shape[-1], f'{q.shape} vs {action.shape}'
            q = tf.reduce_sum(q * action, -1)
        return q

    def reset_noisy(self):
        if self._layer_type == 'noisy':
            if self._duel:
                self._v_head.reset()
            self._a_head.reset()


class DQN(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0, return_stats=False):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims in (2, 4), x.shape

        x = self.encoder(x)
        noisy = not deterministic
        q = self.q(x, noisy=noisy, reset=False)
        action = tf.argmax(q, axis=-1, output_type=tf.int32)
        terms = {}
        if return_stats:
            terms = {'q': q}
        action = epsilon_greedy(action, epsilon,
            is_action_discrete=True, action_dim=self.q.action_dim)
        action = tf.squeeze(action)

        return action, terms


def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    return dict(
        encoder=Encoder(config['encoder'], name='encoder'),
        q=Q(config['q'], action_dim, name='q'),
        target_encoder=Encoder(config['encoder'], name='target_encoder'),
        target_q=Q(config['q'], action_dim, name='target_q'),
    )

def create_model(config, env, **kwargs):
    return DQN(config, env, **kwargs)
