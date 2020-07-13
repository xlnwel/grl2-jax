import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.timer import TBTimer
from core.module import Module
from core.decorator import config
from nn.func import mlp
from nn.layers import Noisy
from nn.func import cnn
        

LSTMState = collections.namedtuple('LSTMState', ['h', 'c'])
class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)
        self._dtype = global_policy().compute_dtype

        self._action_dim = action_dim

        """ Network definition """
        self._cnn = cnn(self._cnn, time_distributed=True)

        # RNN layer
        self._rnn = layers.LSTM(self._lstm_units, return_sequences=True, return_state=True)

        if self._duel:
            self._v_head = mlp(
                self._units_list, 
                out_size=1, 
                activation=self._activation, 
                out_dtype='float32',
                name='v')
        self._a_head = mlp(
            self._units_list, 
            out_size=action_dim, 
            activation=self._activation, 
            out_dtype='float32',
            name='a' if self._duel else 'q')

    @tf.function
    def action(self, x, state, deterministic, epsilon=0, prev_action=None, prev_reward=None):
        qs, state = self.value(x, state, prev_action, prev_reward)
        qs = tf.squeeze(qs)
        action = tf.cast(tf.argmax(qs, axis=-1), tf.int32)
        if deterministic:
            return action, state
        else:
            rand_act = tfd.Categorical(tf.zeros_like(qs)).sample()
            eps_action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)
            prob = tf.cast(eps_action == action, tf.float32)
            prob = prob * (1 - epsilon) + epsilon / self._action_dim
            logpi = tf.math.log(prob)
            return eps_action, {'logpi': logpi}, state
    
    @tf.function
    def value(self, x, state, action=None, prev_action=None, prev_reward=None):
        if self._cnn is None:
            x = tf.reshape(x, (-1, 1, x.shape[-1]))
            if not hasattr(self, 'shared_layers'):
                self.shared_layers = mlp(self._shared_units, activation=self._activation)
        else:
            x = tf.reshape(x, (-1, 1, *x.shape[-3:]))
        x = self.cnn(x)
        x, state = self.rnn(x, state, prev_action, prev_reward)
        q = self.mlp(x, action=action)

        return q, state

    def cnn(self, x):
        if self._cnn:
            x = self._cnn(x)
        return x

    def rnn(self, x, state, prev_action=None, prev_reward=None):
        if prev_action is not None or prev_reward is not None:
            prev_action = tf.one_hot(prev_action, self._action_dim, dtype=x.dtype)
            prev_reward = tf.cast(tf.expand_dims(prev_reward, -1), dtype=x.dtype)
            x = tf.concat([x, prev_action, prev_reward], axis=-1)
        x = self._rnn(x, initial_state=state)
        x, state = x[0], x[1:]
        return x, state

    def mlp(self, x, action=None):
        if self._duel:
            v = self._v_head(x)
            a = self._a_head(x)
            q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        else:
            q = self._a_head(x)

        if action is not None:
            if len(action.shape) < len(q.shape):
                action = tf.one_hot(action, self._action_dim, dtype=q.dtype)
            assert q.shape[1:] == action.shape[1:], f'{q.shape} vs {action.shape}'
            q = tf.reduce_sum(q * action, -1)
            
        return q

    def reset_noisy(self):
        if self._layer_type == 'noisy':
            if self._duel:
                self._v_head.reset()
            self._a_head.reset()

    def reset_states(self, states=None):
        self._rnn.reset_states(states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is None:
            assert batch_size is not None
            inputs = tf.zeros([batch_size, 1, 1])
        if dtype is None:
            dtype = global_policy().compute_dtype
        return tf.nest.map_structure(lambda x: tf.cast(x, dtype), 
                    self._rnn.get_initial_state(inputs))

    @property
    def state_size(self):
        return LSTMState(*self._rnn.cell.state_size)

def create_model(config, env):
    action_dim = env.action_dim
    q = Q(config, action_dim, 'q')
    target_q = Q(config, action_dim, 'target_q')
    return dict(
        q=q,
        target_q=target_q,
    )
