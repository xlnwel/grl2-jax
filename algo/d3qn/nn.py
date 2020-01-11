import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu

from utility.display import pwc
from core.tf_config import build
from nn.func import mlp
from nn.layers import Noisy
from nn.func import cnn
        

class Q(tf.Module):
    def __init__(self, config, state_shape, n_actions, name='q'):
        super().__init__(name=name)

        self.n_actions = n_actions

        # parameters
        cnn_name = config.get('cnn')
        activation = config.get('activation', 'relu')

        """ Network definition """
        if cnn_name:
            self.cnn = cnn(cnn_name)

        self.v_head = mlp(config['v_units'], out_dim=1, layer_type=Noisy, activation=activation, name='v')
        self.a_head = mlp(config['a_units'], out_dim=n_actions, layer_type=Noisy, activation=activation, name='a')

        # build for variable initialization
        TensorSpecs = [
            (state_shape, tf.float32, 'state')
        ]
        self.action = build(self._action, TensorSpecs)
        self.det_action = build(self._det_action, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def _action(self, x):
        if len(x.shape) == 4:
            x = x / 255.
        with tf.name_scope('action'):
            qs = self._step(x, reset=False, noisy=True, name='action')

            action = tf.argmax(qs, axis=1)
            return action
    
    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def _det_action(self, x):
        if len(x.shape) == 4:
            x = x / 255.
        with tf.name_scope('det_action'):
            qs = self._step(x, reset=False, noisy=False, name='det_action')

            action = tf.argmax(qs, axis=1)
            return action

    @tf.Module.with_name_scope
    def train_det_value(self, x, action):
        with tf.name_scope('train_det_qs'):
            qs = self._step(x, False, False, name='train_det_value')
            assert action.shape[1:] == qs.shape[1:], f'action({action.shape}) != qs({qs.shape})'
            q = tf.reduce_sum(action * qs, axis=1, keepdims=True)

        return q

    @tf.Module.with_name_scope
    def train_action(self, x):
        with tf.name_scope('train_det_action'):
            qs = self._step(x, False, False, name='train_action')
            
            return tf.argmax(qs, axis=1)

    @tf.Module.with_name_scope
    def train_value(self, x, action):
        with tf.name_scope('train_step'):
            qs = self._step(x, name='train_value')
            assert action.shape[1:] == qs.shape[1:], f'action({action.shape}) != qs({qs.shape})'
            q = tf.reduce_sum(action * qs, axis=1, keepdims=True)

        return q

    def _step(self, x, reset=True, noisy=True, name=None):
        if hasattr(self, 'cnn'):
            x = self.cnn(x)

        v = self.v_head(x, reset=reset, noisy=noisy)
        
        a = self.a_head(x, reset=reset, noisy=noisy)
        
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)

        return q

    def reset_noisy(self):
        self.v_head.reset()
        self.a_head.reset()

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]


def create_model(model_config, state_shape, n_actions, is_action_discrete=True):
    q_config = model_config['q']
    q = Q(q_config, state_shape, n_actions, 'q')
    target_q = Q(q_config, state_shape, n_actions, 'target_q')
    return dict(
        actor=q,
        q1=q,
        target_q1=target_q,
    )
