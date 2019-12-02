import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utility.display import pwc
from core.tf_config import build
from utility.rl_utils import clip_but_pass_gradient, logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.layers.func import mlp_layers, dnc_rnn
from nn.initializers import get_initializer
from nn.cnn import get_cnn


class PPOAC(tf.Module):
    def __init__(self, 
                config, 
                state_shape, 
                action_dim, 
                is_action_discrete, 
                batch_size, 
                name):
        super().__init__(name=name)
        
        # network parameters
        cnn_name = config.get('cnn')
        shared_mlp_units = config.get('shared_mlp_units')
        use_dnc = config['use_dnc']
        lstm_units = config['lstm_units']
        dnc_config = dict(
            output_size=128,
            access_config=dict(memory_size=64, word_size=32, num_reads=1, num_writes=1),
            controller_config=dict(units=128),
            rnn_config=dict(return_sequences=True, stateful=True)
        )
        actor_units = config['actor_units']
        critic_units = config['critic_units']
        self.rnn_input_size = shared_mlp_units[-1]

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        initializer_name = config.get('kernel_initializer', 'he_uniform')
        kernel_initializer = get_initializer(initializer_name)

        self.batch_size = batch_size

        """ Network definition """
        if cnn_name:
            self.cnn = get_cnn('ftw')
        # shared mlp layers
        if shared_mlp_units:
            self.shared_mlp = mlp_layers(
                shared_mlp_units, 
                norm=norm, 
                activation=activation, 
                kernel_initializer=kernel_initializer()
            )
        
        # RNN layer
        if use_dnc:
            self.rnn = dnc_rnn(**dnc_config)
        else:
            self.rnn = layers.LSTM(lstm_units, return_sequences=True, return_state=True)

        # actor/critic head
        self.actor = mlp_layers(actor_units, 
                                out_dim=action_dim, 
                                norm=norm, 
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
        
        # build for variable initialization
        if use_dnc:
            # fake_inputs = tf.zeros([batch_size, 1, self.rnn_input_size])
            # initial_state = self.rnn.get_initial_state(fake_inputs)
            # self.rnn(inputs=fake_inputs, initial_state=initial_state)
            # self.rnn.reset_states()
            raise NotImplementedError('tf.function requires different initial states for dnc')

        # cannot build since we define initial_state as a list not a tensor
        TensorSpecs = [([None, *state_shape], tf.float32, 'state'),
                        ([lstm_units], tf.float32, 'h'),
                        ([lstm_units], tf.float32, 'c')]
        self.rnn_states = build(self._rnn_states, TensorSpecs, sequential=False, batch_size=batch_size)

    @tf.function
    @tf.Module.with_name_scope
    def step(self, x, initial_state):
        """ Run PPOAC in the real-time mode
        
        Args:
            x: a batch of states of shape `[batch_size, *state_shape]
            initial_state: initial state for LSTM
        Returns: 
            action: actions sampled from the policy distribution of shape
                `[batch_size, action_dim]`
            logpi: the logarithm of the policy distribution of shape
                `[batch_size, 1]`
            value: state values of shape `[batch_size, 1]`
        """
        pwc(f'{self.name} "step" is retracing: x={x}', color='cyan')
        # expand time dimension assuming x has shape `[batch_size, *state_shape]`
        x = tf.expand_dims(x, 1)
        x, states = self._common_layers(x, initial_state)

        mu = self._head(x, self.actor)
        value = self._head(x, self.critic)

        policy_distribution = self.PolicyDistributionType((mu, self.logstd))

        raw_action = policy_distribution.sample()
        logpi = policy_distribution.logp(tf.stop_gradient(raw_action))

        # squash action
        action = tf.tanh(raw_action)
        logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)
        
        return tf.squeeze(action, 1), tf.squeeze(logpi, 1), tf.squeeze(value, 1), states

    @tf.function
    @tf.Module.with_name_scope
    def det_action(self, x, initial_state):
        """ Get the deterministic actions given state x 
        
        Args:
            x: a batch of states of shape `[batch_size, *state_shape]
            initial_state: initial state for LSTM
        Returns:
            determinitistic action of shape `[batch_size, action_dim]`
        """
        pwc(f'{self.name} "det_action" is retracing: x={x}', color='cyan')
        with tf.name_scope('det_action'):
            x = tf.expand_dims(x, 1)
            x, states = self._common_layers(x, initial_state)
            action = self._head(x, self.actor)

            return tf.squeeze(tf.tanh(action), 1), states
    
    @tf.function
    @tf.Module.with_name_scope
    def _rnn_states(self, x, h, c):
        pwc(f'{self.name} "rnn_states" is retracing', color='cyan')
        _, states = self._common_layers(x, [h, c])

        return states
        
    def train_step(self, x, a, initial_state):
        """ Run PPOAC in the training mode
        
        Args:
            x: a batch of states of shape `[batch_size, steps, *state_shape]`
            a: a batch of actions of shape `[batch_size, steps, action_dim]`
            initial_state: initial state for LSTM
        Returns: 
            action: actions sampled from the policy distribution of shape
                `[batch_size, steps, action_dim]`
            logpi: the logarithm of the policy distribution of shape
                `[batch_size, steps, 1]`
            value: state values of shape `[batch_size, steps, 1]`
        """
        pwc(f'{self.name} "train_step" is retracing: x={x}, a={a}', color='cyan')
        with tf.name_scope('train_step'):
            x, states = self._common_layers(x, initial_state)

            mu = self._head(x, self.actor)
            value = self._head(x, self.critic)

            policy_distribution = DiagGaussian((mu, self.logstd))
            # correction for squashed action
            # clip_but_pass_gradient is used to avoid case when a == -1, 1
            raw_action = tf.math.atanh(clip_but_pass_gradient(a, -1+1e-7, 1-1e-7))
            logpi = policy_distribution.logp(tf.stop_gradient(raw_action))
            logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)

            entropy = policy_distribution.entropy()

            return logpi, entropy, value, states

    def _common_layers(self, x, initial_state):
        if hasattr(self, 'cnn'):
            x = self.cnn(x)
        if hasattr(self, 'shared_mlp'):
            for l in self.shared_mlp:
                x = l(x)

        x, h, c = self.rnn(x, initial_state=initial_state)
        
        return x, [h, c]

    def _head(self, x, layers):
        for l in layers:
            x = l(x)
        
        return x

    def reset_states(self, states=None):
        self.rnn.reset_states(states)

    def get_initial_state(self):
        """ Get the initial states of rnn, 
        should only be called after the model is built """
        fake_inputs = tf.zeros([self.batch_size, 1, self.rnn_input_size])
        return self.rnn.get_initial_state(fake_inputs)


def create_model(model_config, state_shape, action_dim, is_action_discrete, n_envs):
    ac = PPOAC(
        model_config, 
        state_shape, 
        action_dim, 
        is_action_discrete,
        n_envs,
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
    state_shape = [5]
    action_dim = 1
    batch_size = 3
    is_action_discrete = False
    seq_len = 2
    ac = PPOAC(config, state_shape, action_dim, is_action_discrete, batch_size, 'ac')

    from utility.display import display_var_info

    display_var_info(ac.trainable_variables)

    # for _ in range(2):
    #     pwc(ac.rnn[0].states)
    #     x = np.random.rand(3, 1, 4)
    #     y = ac.value(x)
    # # ac.reset_states()
    # test rnn
    x = np.random.rand(batch_size, seq_len, *state_shape)
    states = ac.get_initial_state()
    np.testing.assert_allclose(states, 0.)
    for i in range(seq_len):
        _, _, _, states = ac.step(tf.convert_to_tensor(x[:, i], tf.float32), states)
    step_states = states
    states = ac.get_initial_state()
    np.testing.assert_allclose(states, 0.)

    a = np.random.rand(batch_size, seq_len, action_dim)
    _, _, _, states = ac.train_step(tf.convert_to_tensor(x, tf.float32), 
        tf.convert_to_tensor(a, tf.float32), states)
    train_step_states = states
    np.testing.assert_allclose(step_states, train_step_states)

