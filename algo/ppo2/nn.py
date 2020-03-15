import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from core.module import Module
from core.tf_config import build
from core.decorator import config
from utility.rl_utils import clip_but_pass_gradient, logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical, TanhBijector
from nn.func import cnn, mlp, dnc_rnn
from nn.utils import get_initializer


class PPOAC(Module):
    @config
    def __init__(self, obs_shape, action_dim, is_action_discrete, name):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        
        # network parameters
        self._cnn_name = None if self._cnn_name.lower() == 'none' else self._cnn_name
        
        """ Network definition """
        if self._cnn_name:
            self._shared_layers = cnn(self._cnn_name, time_action_distributed=True)
        # shared mlp layers
        if self._shared_mlp_units:
            self._shared_layers = mlp(
                self._shared_mlp_units, 
                norm=self._norm, 
                activation=self._activation, 
                kernel_initializer=get_initializer('orthogonal', gain=.01))
        # RNN layer
        if self._use_dnc:
            dnc_config = dict(
                output_size=128,
                access_config=dict(memory_size=64, word_size=32, num_reads=1, num_writes=1),
                controller_config=dict(units=128),
                rnn_config=dict(return_sequences=True, return_state=True))
            self.rnn = dnc_rnn(**dnc_config)
        else:
            self.rnn = layers.LSTM(self._lstm_units, return_sequences=True, return_state=True)

        # actor/critic head
        self.actor = mlp(self._actor_units, 
                        out_dim=action_dim, 
                        norm=self._norm,
                        name='actor',
                        activation=self._activation, 
                        kernel_initializer=get_initializer('orthogonal', gain=.01))
        if not self._is_action_discrete:
            self.logstd = tf.Variable(initial_value=np.zeros(action_dim), 
                                        dtype=tf.float32, 
                                        trainable=True, 
                                        name=f'actor/logstd')
        self.critic = mlp(self._critic_units, 
                            out_dim=1,
                            norm=self._norm,
                            name='critic', 
                            activation=self._activation, 
                            kernel_initializer=get_initializer('orthogonal', gain=1.))

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def step(self, x, state):
        pwc(f'{self.name} "step" is retracing: x={x.shape}', color='cyan')
        # assume x is of shape `[batch_size, *obs_shape]`
        x = tf.expand_dims(x, 1)
        action_dist, value, state = self.train_step(x, state)
        action = action_dist.sample()
        logpi = tf.squeeze(action_dist.log_prob(action), 1)
        action = tf.squeeze(action, 1)
        value = tf.squeeze(value, 1)

        return action, logpi, value, state

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def det_action(self, x, state):
        pwc(f'{self.name} "det_action" is retracing: x={x.shape}', color='cyan')
        with tf.name_scope('det_action'):
            # assume x is of shape [batch_size, *obs_shape]
            x = tf.expand_dims(x, 1)
            x, state = self._common_layers(x, state)
            x = tf.squeeze(x, 1)

            actor_output = self.actor(x)
            assert actor_output.shape.ndims == 2

            if self._is_action_discrete:
                return tf.argmax(actor_output, -1), state
            else:
                return tf.tanh(actor_output), state
    
    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def rnn_state(self, x, h, c):
        pwc(f'{self.name} "rnn_state" is retracing', color='cyan')
        _, state = self._common_layers(x, [h, c])

        return state
        
    def train_step(self, x, state):
        pwc(f'{self.name} "train_step" is retracing: x={x.shape}', color='cyan')
        with tf.name_scope('train_step'):
            x, state = self._common_layers(x, state)

            actor_output = self.actor(x)
            value = tf.squeeze(self.critic(x), -1)
            if self._is_action_discrete:
                action_dist = tfd.Categorical(actor_output)
            else:
                action_dist = DiagGaussian(actor_output, self.logstd)
                # action_dist = tfd.TransformedDistribution(action_dist, TanhBijector())
                # action_dist = tfd.Independent(action_dist, 1)

            return action_dist, value, state

    def _common_layers(self, x, state):
        if hasattr(self, 'cnn'):
            x = tf.cast(x, tf.float32)
            x = x / 255.
        if hasattr(self, '_shared_layers'):
            x = self._shared_layers(x)

        x = self.rnn(x, initial_state=state)
        x, state = x[0], x[1:]
        return x, state

    def reset_state(self, state=None):
        self.rnn.reset_state(state)

    def get_initial_state(self, inputs=None, batch_size=None):
        """ Get the initial state of rnn, 
        should only be called after the model is built """
        if inputs is None:
            assert batch_size is not None
            inputs = tf.zeros([batch_size, 1, 1])
        return self.rnn.get_initial_state(inputs)


def create_model(model_config, obs_shape, action_dim, is_action_discrete, n_envs):
    ac = PPOAC(
        model_config, 
        obs_shape,
        action_dim, 
        is_action_discrete,
        'ac')

    return dict(ac=ac)

if __name__ == '__main__':
    config = dict(
        shared_mlp_units=[4],
        use_dnc=False,
        lstm_units=3,
        actor_units=[2],
        critic_units=[2],
        norm='none',
        activation='relu',
        kernel_initializer='he_uniform'
    )

    batch_size = np.random.randint(1, 10)
    seq_len = np.random.randint(1, 10)
    obs_shape = [5]
    action_dim = np.random.randint(1, 10)
    for is_action_discrete in [True, False]:
        action_dtype = np.int32 if is_action_discrete else np.float32
        
        ac = PPOAC(config, obs_shape, action_dim, is_action_discrete, batch_size, 'ac')

        from utility.display import display_var_info

        display_var_info(ac.trainable_variables)

        # test rnn state
        x = np.random.rand(batch_size, seq_len, *obs_shape)
        state = ac.get_initial_state()
        np.testing.assert_allclose(state, 0.)
        for i in range(seq_len):
            _, _, _, state = ac.step(tf.convert_to_tensor(x[:, i], tf.float32), state)
        step_state = state
        state = ac.get_initial_state()
        np.testing.assert_allclose(state, 0.)

        if is_action_discrete:
            a = np.random.randint(low=0, high=action_dim, size=(batch_size, seq_len))
        else:
            a = np.random.rand(batch_size, seq_len, action_dim)
        _, state = ac._common_layers(tf.convert_to_tensor(x, tf.float32), state)
        train_step_state = state
        np.testing.assert_allclose(step_state, train_step_state, atol=1e-5, rtol=1e-5)

