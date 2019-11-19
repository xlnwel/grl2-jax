import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utility.display import pwc
from utility.rl_utils import clip_but_pass_gradient, logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from utility.logger import Logger
from nn.layers.func import mlp_layers, dnc_rnn
from nn.initializers import get_initializer


class PPOAC(keras.Model):
    def __init__(self, state_shape, action_dim, is_action_discrete, batch_size, name, **kwargs):
        super().__init__(name=name)
        
        # network parameters
        shared_mlp_units = [128]
        use_dnc = False
        lstm_units = 128
        dnc_config = dict(
            output_size=128,
            access_config=dict(memory_size=64, word_size=32, num_reads=1, num_writes=1),
            controller_config=dict(units=128),
            rnn_config=dict(return_sequences=True, stateful=True)
        )
        actor_units = [64]
        critic_units = [128]

        norm = None
        activation = 'relu'
        kernel_initializer = get_initializer('he')

        self.batch_size = batch_size

        """ network """
        # shared mlp layers
        self.shared_mlp = mlp_layers(shared_mlp_units, 
                                    norm=norm, 
                                    name='shared_mlp',
                                    activation=activation, 
                                    kernel_initializer=kernel_initializer())

        # RNN layer
        if use_dnc:
            self.rnn = dnc_rnn(**dnc_config)
        else:
            self.rnn = layers.LSTM(lstm_units, return_sequences=True, stateful=True)

        # actor/critic head
        self.actor = mlp_layers(actor_units, 
                                out_dim=action_dim, 
                                norm=norm, 
                                name='actor', 
                                activation=activation, 
                                kernel_initializer=kernel_initializer())
        self.logstd = tf.Variable(initial_value=np.zeros(action_dim), 
                                    dtype=tf.float32, 
                                    trainable=True, 
                                    name=f'actor/logstd')
        self.critic = mlp_layers(critic_units, 
                                out_dim=1,
                                norm=norm, 
                                name='critic', 
                                activation=activation, 
                                kernel_initializer=kernel_initializer())

        # policy distribution type
        self.PolicyDistributionType = Categorical if is_action_discrete else DiagGaussian
        
        # Fake call for variable initialization
        if use_dnc:
            # this has to be done for DNC, for some obscure reason, we have to pass
            # in initial_state for the first time so that the RNN can know a 
            # high-order state_size
            tf.summary.experimental.set_step(0)
            input_size = shared_mlp_units[-1]
            fake_inputs = tf.zeros([batch_size, 1, input_size])
            initial_state = self.rnn.get_initial_state(fake_inputs)
            self.rnn(inputs=fake_inputs, initial_state=initial_state)
            self.rnn.reset_states()
        self(tf.keras.Input(shape=state_shape, batch_size=batch_size))

    @tf.function
    def call(self, x):
        """ Run PPOAC in the real-time mode
        
        Args:
            x: a batch of states of shape `[batch_size, *state_shape]
        Returns: 
            action: actions sampled from the policy distribution of shape
                `[batch_size, action_dim]`
            logpi: the logarithm of the policy distribution of shape
                `[batch_size, 1]`
            value: state values of shape `[batch_size, 1]`
        """
        pwc(f'AC call is retracing: x={x}', color='cyan')
        # expand time dimension assuming x has shape `[batch_size, *state_shape]`
        x = tf.expand_dims(x, 1)
        with tf.name_scope('call'):
            x = self._common_layers(x)

            mu = self._head(x, self.actor)
            value = self._head(x, self.critic)

            policy_distribution = self.PolicyDistributionType((mu, self.logstd))

            raw_action = policy_distribution.sample()
            logpi = policy_distribution.logp(tf.stop_gradient(raw_action))
    
            # squash action
            action = tf.tanh(raw_action)
            logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)
            
            return tf.squeeze(action, 1), tf.squeeze(logpi, 1), tf.squeeze(value, 1)

    @tf.function
    def det_action(self, x):
        """ Get the deterministic actions given state x 
        
        Args:
            x: a batch of states of shape `[batch_size, *state_shape]
        Returns:
            determinitistic action of shape `[batch_size, action_dim]`
        """
        pwc(f'AC det_action is retracing: x={x}', color='cyan')
        x = tf.expand_dims(x, 1)
        with tf.name_scope('det_action'):
            x = self._common_layers(x)
            action = self._head(x, self.actor)

            return tf.squeeze(tf.tanh(action), 1)
    
    def train_step(self, x, a):
        """ Run PPOAC in the training mode
        
        Args:
            x: a batch of states of shape `[batch_size, steps, *state_shape]
        Returns: 
            action: actions sampled from the policy distribution of shape
                `[batch_size, steps, action_dim]`
            logpi: the logarithm of the policy distribution of shape
                `[batch_size, steps, 1]`
            value: state values of shape `[batch_size, steps, 1]`
        """
        pwc(f'AC logpi_and_entropy is retracing: x={x}, a={a}', color='cyan')
        with tf.name_scope('train_step'):
            x = self._common_layers(x)

            mu = self._head(x, self.actor)
            value = self._head(x, self.critic)

            policy_distribution = DiagGaussian((mu, self.logstd))
            # correction for squashed action
            # clip_but_pass_gradient is used to avoid case when a == -1, 1
            raw_action = tf.math.atanh(clip_but_pass_gradient(a, -1+1e-7, 1-1e-7))
            logpi = policy_distribution.logp(tf.stop_gradient(raw_action))
            logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)

            entropy = policy_distribution.entropy()

            return logpi, entropy, value

    def _common_layers(self, x):
        pwc(f'AC common layers is retracing', color='cyan')
        
        for l in self.shared_mlp:
            x = l(x)

        x = self.rnn(x)
        
        return x

    def _head(self, x, layers):
        for l in layers:
            x = l(x)
        
        return x


if __name__ == '__main__':
    ac = PPOAC([4], 2, False, 3, 'ac')
    # to print proper summary, comment out @tf.function related to call
    # ac.summary()

    # from utility.display import display_var_info

    # display_var_info(ac.trainable_weights)

    # for _ in range(2):
    #     pwc(ac.rnn[0].states)
    #     x = np.random.rand(3, 1, 4)
    #     y = ac.value(x)
    # # ac.reset_states()
    x = np.random.rand(3, 2, 4)
    pwc(ac.rnn.states)
    for i in range(2):
        y = ac(x[:, i:i+1])
    pwc(ac.rnn.states)
    ac.reset_states()
    pwc(ac.rnn.states)
    ac(x)
    pwc(ac.rnn.states)
