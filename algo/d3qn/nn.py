import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu

from utility.display import pwc
from core.tf_config import build
from nn.layers.func import mlp_layers
from nn.initializers import get_initializer
from nn.layers.noisy import Noisy
from nn.cnn import get_cnn
        

class Q(tf.Module):
    def __init__(self, config, state_shape, n_actions, name='q'):
        super().__init__(name=name)

        self.n_actions = n_actions

        # parameters
        cnn_name = config.get('cnn')

        """ Network definition """
        if cnn_name:
            self.cnn = get_cnn(cnn_name)
        self.v_head = [
            Noisy(256),
            Noisy(1),
        ]
        self.a_head = [
            Noisy(256),
            Noisy(n_actions),
        ]

        # build for variable initialization
        TensorSpecs = [
            (state_shape, tf.float32, 'state')
        ]
        self.det_action = build(self._det_action, TensorSpecs)
    
    @tf.Module.with_name_scope
    def train_det_value(self, x, action):
        with tf.name_scope('train_det_qs'):
            qs = self._step(x, False, False, name='train_det_value')

            q = tf.reduce_sum(
                    tf.one_hot(action, self.n_actions) * qs, 
                    axis=1, keepdims=True)

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

            q = tf.reduce_sum(
                    tf.one_hot(action, self.n_actions) * qs, 
                    axis=1, keepdims=True)

        return q

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def action(self, x):
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
            
    def _step(self, x, reset=True, noisy=True, name=None):
        print(name)
        if hasattr(self, 'cnn'):
            x = self.cnn(x)

        v = x
        v = self.v_head[0](v, reset=reset, noisy=noisy)
        v = relu(v)
        v = self.v_head[1](v, reset=reset, noisy=noisy)
        
        a = x
        a = self.a_head[0](a, reset=reset, noisy=noisy)
        a = relu(a)
        a = self.a_head[1](a, reset=reset, noisy=noisy)
        
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)

        return q

    def reset_noisy(self):
        for l in self.v_head + self.a_head:
            l.reset()

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
