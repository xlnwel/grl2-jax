import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import global_policy
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.timer import TBTimer
from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from nn.layers import Noisy
from nn.func import cnn
        

LSTMState = collections.namedtuple('LSTMState', ['h', 'c'])
class Encoder(Module):
    def __init__(self, config, name='encoder'):
        super().__init__(name=name)
        config = config.copy()
        config['time_distributed'] = True
        self._cnn = cnn(**config)
    
    def call(self, x):
        x = self._layers(x)
        return x


class RNN(Module):
    @config
    def __init__(self, name='rnn'):
        super().__init__(name=name)
        self._rnn = layers.LSTM(self._lstm_units, return_sequences=True, return_state=True)

    def call(self, x, state, prev_action=None, prev_reward=None):
        assert x.shape.ndims == 3, x.shape
        if self._additional_input:
            prev_reward = tf.cast(tf.expand_dims(prev_reward, axis=-1), dtype=x.dtype)
            assert x.shape.ndims == prev_action.shape.ndims, (x.shape, prev_action.shape)
            assert x.shape.ndims == prev_reward.shape.ndims, (x.shape, prev_reward.shape)
            x = tf.concat([x, prev_action, prev_reward], axis=-1)
        x = self._rnn(x, initial_state=state)
        x, state = x[0], LSTMState(*x[1:])
        return x, state


class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)

        self.action_dim = action_dim

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
    
    @property
    def action_dim(self):
        return self.action_dim

    def call(self, x, action=None):
        if self._duel:
            v = self._v_head(x)
            a = self._a_head(x)
            q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        else:
            q = self._a_head(x)

        if action is not None:
            if len(action.shape) < len(q.shape):
                action = tf.one_hot(action, self.action_dim, dtype=q.dtype)
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


class RDQN(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, state, evaluation, epsilon=0, prev_action=None, prev_reward=None):
        assert x.shape.ndims == 4, x.shape
        # add time dimension
        x = tf.expand_dims(x, axis=1)
        prev_action = tf.reshape(prev_action, (-1, 1))
        prev_action = tf.one_hot(prev_action, self.q.action_dim, dtype=x.dtype)
        prev_reward = tf.reshape(prev_reward, (-1, 1))
        assert x.shape.ndims == 5, x.shape
        assert prev_action.shape.ndims == 2, x.shape
        assert prev_reward.shape.ndims == 2, x.shape
        x = self.encoder(x)
        x, state = self.rnn(x, state, prev_action, prev_reward)
        qs, state = self.q(x)
        action = tf.argmax(qs, axis=-1, output_type=tf.int32)
        action = tf.squeeze(action)
        q = tf.math.reduce_max(qs, -1)
        q = tf.squeeze(q)
        if epsilon > 0:
            rand_act = tf.random.uniform(
                action.shape, 0, self.q.action_dim, dtype=tf.int32)
            eps_action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)
            prob = tf.cast(eps_action == action, tf.float32)
            prob = prob * (1. - epsilon) + epsilon / self.q.action_dim
            logpi = tf.math.log(prob)
            action = eps_action
        else:
            logpi = tf.zeros_like(q)
        return action, {'logpi': logpi, 'q': q}, state


def create_components(config, env):
    action_dim = env.action_dim
    encoder_config = config['encoder']
    rnn_config = config['rnn']
    q_config = config['q']
    return dict(
        encoder=Encoder(encoder_config, name='encoder'),
        rnn=RNN(rnn_config, action_dim, name='rnn'),
        q=Q(q_config, action_dim, 'q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_rnn=RNN(rnn_config, name='target_rnn'),
        target_q=Q(q_config, action_dim, 'target_q'),
    )

def create_model(config, env, **kwargs):
    return RDQN(config, env, **kwargs)
