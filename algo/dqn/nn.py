import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp, cnn
from nn.layers import Noisy
        

class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)
        self._dtype = global_policy().compute_dtype

        self.action_dim = action_dim

        """ Network definition """
        kwargs = {}
        if hasattr(self, '_kernel_initializer'):
            kwargs['kernel_initializer'] = self._kernel_initializer
        self._cnn = cnn(self._cnn_name, out_size=self._cnn_out_size, **kwargs)

        layer_type = dict(noisy=Noisy, dense=layers.Dense)[self._layer_type]
        if self._duel:
            self._v_head = mlp(
                self._units_list, 
                out_size=1, 
                layer_type=layer_type, 
                activation=self._activation, 
                out_dtype='float32',
                name='v',
                **kwargs)
        self._a_head = mlp(
            self._units_list, 
            out_size=action_dim, 
            layer_type=layer_type, 
            activation=self._activation, 
            out_dtype='float32',
            name='a' if self._duel else 'q',
            **kwargs)

    @tf.function
    def action(self, x, noisy=True, reset=True):
        q = self.value(x, noisy=noisy, reset=reset)
        return tf.argmax(q, axis=-1, output_type=tf.int32)
    
    @tf.function
    def value(self, x, action=None, noisy=True, reset=True):
        x = self.cnn(x)
        q = self.mlp(x, action=action, noisy=noisy, reset=reset)
        return q

    def cnn(self, x):
        if self._cnn:
            x = self._cnn(x)
        return x

    def mlp(self, x, action=None, noisy=True, reset=True):
        if self._duel:
            v = self._v_head(x, noisy=noisy, reset=reset)
            a = self._a_head(x, noisy=noisy, reset=reset)
            q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        else:
            q = self._a_head(x, noisy=noisy, reset=reset)

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
    def action(self, x, deterministic=False, epsilon=0):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims in (2, 4), x.shape

        noisy = not deterministic
        q = self.q.value(x, noisy=noisy, reset=False)
        action = tf.argmax(q, axis=-1, output_type=tf.int32)
        if epsilon > 0:
            rand_act = tf.random.uniform(
                action.shape, 0, self.q.action_dim, dtype=tf.int32)
            action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)
        action = tf.squeeze(action)

        return action, {'q': q}


def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    q = Q(config, action_dim, name='q')
    target_q = Q(config, action_dim, name='target_q')
    return dict(
        q=q,
        target_q=target_q,
    )

def create_model(config, env, **kwargs):
    return DQN(config, env, **kwargs)
