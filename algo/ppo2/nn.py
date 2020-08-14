import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import cnn, mlp
from nn.rnn import LSTMCell, LSTMState
from algo.ppo.nn import Encoder, Actor, Critic


class RNN(Module):
    @config
    def __init__(self, name='rnn'):
        super().__init__(name=name)
        cell = LSTMCell(self._lstm_units, use_ln=self._lstm_ln)
        self._rnn = layers.RNN(cell, return_sequences=True, return_state=True)
    
    def __call__(self, x, state, mask=None, prev_action=None):
        if self._additional_input:
            assert x.shape.ndims == prev_action.shape.ndims, (x.shape, prev_action.shape)
            x = tf.concat([x, prev_action], axis=-1)
        mask = mask[..., None]
        assert len(x.shape) == len(mask.shape), f'x({x.shape}), mask({mask.shape})'
        x = self._rnn((x, mask), initial_state=state)
        x, state = x[0], LSTMState(*x[1:])
        return x, state

    def reset_states(self, states=None):
        self._rnn.reset_states(states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is None:
            assert batch_size is not None
            inputs = tf.zeros([batch_size, 1, 1])
        if dtype is None:
            dtype = global_policy().compute_dtype
        return LSTMState(*tf.nest.map_structure(lambda x: tf.cast(x, dtype), 
                    self._rnn.get_initial_state(inputs)))

    @property
    def state_size(self):
        return self._rnn.cell.state_size

    @property
    def state_keys(self):
        return ['h', 'c']


class PPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)
    
    @tf.function
    def action(self, x, state, mask, deterministic=False, epsilon=0, prev_action=None):
        assert x.shape.ndims % 2 == 0, x.shape
        # add time dimension
        x = tf.expand_dims(x, 1)
        mask = tf.expand_dims(mask, 1)
        if self.rnn._additional_input:
            if self.actor.is_action_discrete:
                prev_action = tf.reshape(prev_action, (-1, 1))
                prev_action = tf.one_hot(prev_action, self.actor.action_dim, dtype=x.dtype)
            else:
                prev_action = tf.reshape(prev_action, (-1, 1, self.actor.action_dim))
        x = self.encoder(x)
        x, state = self.rnn(x, state, mask, prev_action)
        if deterministic:
            act_dist = self.actor(x)
            action = tf.squeeze(act_dist.mode(), 1)
            return action, state
        else:
            act_dist = self.actor(x)
            value = self.critic(x)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            # intend to keep the batch dimension for later use
            out = tf.nest.map_structure(lambda x: tf.squeeze(x, 1), (action, terms))
            return out, state

    def reset_states(self, states=None):
        self.rnn.reset_states(states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.rnn.get_initial_state(inputs, batch_size=batch_size, dtype=dtype)

    @property
    def state_size(self):
        return self.rnn.state_size
    
    @property
    def state_keys(self):
        return self.rnn.state_keys

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    config['encoder']['time_distributed'] = True
    return dict(
        encoder=Encoder(config['encoder']), 
        rnn=RNN(config['rnn']),
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        critic=Critic(config['critic'])
    )

def create_model(config, env, **kwargs):
    return PPO(config, env, **kwargs)
