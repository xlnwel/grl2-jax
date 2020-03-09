import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

from core.tf_config import build
from utility.rl_utils import logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.func import mlp

RSSMState = collections.namedtuple('RSSMState', ('mean', 'std', 'stoch', 'deter'))

class RSSMCell(tf.Module):
    def __init__(self, stoch_size, deter_size, hidden_size, activation=tf.nn.relu, name='rssm'):
        super.__init__(name)

        self._stoch_size = stoch_size
        self._deter_size = deter_size

        self.embed_h = layers.Dense(hidden_size, activation=activation)
        self.img_h = mlp([hidden_size], out_dim=2*stoch_size, activation=activation)
        self.obs_h = mlp([hidden_size], out_dim=2*stoch_size, activation=activation)
        self.cell = layers.GRUCell(self._deter_size)
        
    def observe(self, embed, action, state=None):
        if state is None:
            state = self.get_initial_state(batch_size=tf.shape(action)[0])
        state

    def imagine(self, action, state=None):
        if state is None:
            state = self.get_initial_state(batch_size=tf.shape(action)[0])

    def obs_step(self, prev_state, prev_action, embed):
        x, deter = self._compute_deter_state(prev_state, prev_action)
        x = tf.concat([x, embed], -1)
        x = self.obs_h(x)
        post = self._compute_rssm_state(x, deter)
        return post

    def img_step(self, prev_state, prev_action):
        x, deter = self._compute_deter_state(prev_state, prev_action)
        x = self.img_h(x)
        prior = self._compute_rssm_state(x, deter)
        return prior

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        if inputs:
            assert batch_size is None or batch_size == inputs.shape[0]
            batch_size = inputs.shape[0]
        return RSSMState(mean=tf.zeros([batch_size, self._stoch_size], dtype=dtype),
                        std=tf.zeros([batch_size, self._stoch_size], dtype=dtype),
                        stoch=tf.zeros([batch_size, self._stoch_size], dtype=dtype),
                        deter=self._cell.get_initial_state(batch_size=batch_size, dtype=dtype))
        
    def _compute_deter_state(self, prev_state, prev_action):
        x = tf.concat(prev_state.stoch, prev_action)
        x = self.embed_h(x)
        x, deter = self.cell(x, [prev_state.deter])
        return x, deter[0]  # Keras wraps the state in a list

    def _compute_rssm_state(self, x, deter):
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + .1
        stoch = tfp.distributions.MultivariateNormalDiag(mean, std).sample()
        state = RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)
        return state