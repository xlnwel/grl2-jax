import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from core.module import Module
from core.decorator import config
from nn.func import mlp
from nn.layers import Noisy
from nn.func import cnn
        

class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)
        self._dtype = global_policy().compute_dtype

        self._action_dim = action_dim

        """ Network definition """
        self._cnn = cnn(self._cnn)

        self._v_head = mlp(
            self._v_units, 
            out_dim=1, 
            layer_type=Noisy, 
            activation=self._activation, 
            name='v')
        self._a_head = mlp(
            self._a_units, 
            out_dim=action_dim, 
            layer_type=Noisy, 
            activation=self._activation, 
            name='a')

    def __call__(self, x, deterministic=False, epsilon=0):
        x = np.array(x)
        if not deterministic and np.random.uniform() < epsilon:
            size = x.shape[0] if len(x.shape) % 2 == 0 else None
            return np.random.randint(self._action_dim, size=size)
        if x.dtype == np.uint8:
            x = tf.cast(x, self._dtype) / 255.
        if len(x.shape) % 2 != 0:
            x = tf.expand_dims(x, 0)
        
        noisy = not deterministic
        action = self.action(x, noisy=noisy, reset=True)
        action = np.squeeze(action.numpy())

        return action

    @tf.function(experimental_relax_shapes=True)
    def action(self, x, noisy=True, reset=True):
        q = self.value(x, noisy=noisy, reset=noisy)
        return tfd.Categorical(q).mode()
    
    @tf.function(experimental_relax_shapes=True)
    def value(self, x, action=None, noisy=True, reset=True):
        # tf.debugging.assert_greater_equal(x, 0.)
        # tf.debugging.assert_less_equal(x, 1.)
        if self._cnn:
            x = self._cnn(x)

        v = self._v_head(x, reset=reset, noisy=noisy)
        a = self._a_head(x, reset=reset, noisy=noisy)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)

        if action is not None:
            q = tf.reduce_mean(q * action, -1)
        return q

    def reset_noisy(self):
        self._v_head.reset()
        self._a_head.reset()


def create_model(model_config, action_dim):
    q_config = model_config['q']
    q = Q(q_config, action_dim, 'q')
    target_q = Q(q_config, action_dim, 'target_q')
    return dict(
        q=q,
        target_q=target_q,
    )
