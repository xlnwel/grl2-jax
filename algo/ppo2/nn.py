import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module
from core.decorator import config
from utility.tf_distributions import DiagGaussian, Categorical, TanhBijector
from nn.func import cnn, mlp


class PPOAC(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        
        """ Network definition """
        self._cnn_name = (None if isinstance(self._cnn_name, str) 
            and self._cnn_name.lower() == 'none' else self._cnn_name)
        if self._cnn_name:
            self._shared_layers = cnn(self._cnn_name, time_distributed=True)
        elif self._shared_mlp_units:
            self._shared_layers = mlp(
                self._shared_mlp_units, 
                norm=self._norm, 
                activation=self._activation, 
                kernel_initializer='orthogonal')
        else:
            self._shared_layers = lambda x: x
        # RNN layer
        self._rnn = layers.LSTM(self._lstm_units, return_sequences=True, return_state=True)

        # actor/critic head
        self.actor = mlp(self._actor_units, 
                        out_dim=action_dim, 
                        norm=self._norm,
                        name='actor',
                        activation=self._activation, 
                        kernel_initializer='orthogonal')
        if not self._is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype=global_policy().compute_dtype, 
                trainable=True, 
                name=f'actor/logstd')
        self.critic = mlp(self._critic_units, 
                            out_dim=1,
                            norm=self._norm,
                            name='critic', 
                            activation=self._activation, 
                            kernel_initializer='orthogonal')

    def __call__(self, x, state, return_value=False):
        print(f'{self.name} is retracing: x={x.shape}')
        x = self._shared_layers(x)
        x = self._rnn(x, initial_state=state)
        x, state = x[0], x[1:]
        actor_out = self.actor(x)

        if self._is_action_discrete:
            act_dist = tfd.Categorical(actor_out)
        else:
            act_dist = tfd.MultivariateNormalDiag(actor_out, tf.exp(self.logstd))
        
        if return_value:
            value = tf.squeeze(self.critic(x), -1)
            return act_dist, value, state
        else:
            return act_dist, state

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


def create_model(model_config, action_dim, is_action_discrete, n_envs):
    ac = PPOAC(model_config, action_dim, is_action_discrete, 'ac')

    return dict(ac=ac)
