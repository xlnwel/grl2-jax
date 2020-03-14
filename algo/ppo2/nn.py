import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utility.display import pwc
from core.tf_config import build
from utility.rl_utils import clip_but_pass_gradient, logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.func import cnn, mlp, dnc_rnn
from nn.utils import get_initializer


class PPOAC(tf.Module):
    def __init__(self, 
                config, 
                state_shape, 
                state_dtype,
                action_dim, 
                is_action_discrete, 
                batch_size, 
                name):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        self.batch_size = batch_size
        
        # network parameters
        cnn_name = config.get('cnn')
        shared_mlp_units = config.get('shared_mlp_units')
        use_dnc = config['use_dnc']
        lstm_units = config['lstm_units']
        dnc_config = dict(
            output_size=128,
            access_config=dict(memory_size=64, word_size=32, num_reads=1, num_writes=1),
            controller_config=dict(units=128),
            rnn_config=dict(return_sequences=True, return_state=True)
        )
        actor_units = config['actor_units']
        critic_units = config['critic_units']

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        kernel_initializer = config.get('kernel_initializer', 'orthogonal')
        kernel_initializer = get_initializer(kernel_initializer)

        """ Network definition """
        if cnn_name:
            self.cnn = cnn(cnn_name, time_distributed=True, batch_size=self.batch_size, 
                            kernel_initializer=kernel_initializer)
        # shared mlp layers
        if shared_mlp_units:
            self.shared_mlp = mlp(
                shared_mlp_units, 
                norm=norm, 
                activation=activation, 
                kernel_initializer=kernel_initializer
            )
        if not cnn_name and not shared_mlp_units:
            shared_mlp_units = state_shape
        # RNN layer
        self.rnn_input_size = self.cnn.out_size if cnn_name else shared_mlp_units[-1]
        if use_dnc:
            self.rnn = dnc_rnn(**dnc_config)
        else:
            self.rnn = layers.LSTM(lstm_units, return_sequences=True, return_state=True,
                                    kernel_initializer=kernel_initializer)

        # actor/critic head
        self.actor = mlp(actor_units, 
                        out_dim=action_dim, 
                        norm=norm, 
                        name='actor',
                        activation=activation, 
                        kernel_initializer=get_initializer('orthogonal', gain=.01))
        if not self._is_action_discrete:
            self.logstd = tf.Variable(initial_value=np.zeros(action_dim), 
                                        dtype=tf.float32, 
                                        trainable=True, 
                                        name=f'actor/logstd')
        self.critic = mlp(critic_units, 
                            out_dim=1,
                            norm=norm, 
                            name='critic', 
                            activation=activation, 
                            kernel_initializer=get_initializer('orthogonal', gain=1.))

        # policy distribution type
        self.ActionDistributionType = Categorical if self._is_action_discrete else DiagGaussian
        
        # build for variable initialization
        if use_dnc:
            # fake_inputs = tf.zeros([self.batch_size, 1, self.rnn_input_size])
            # initial_state = self.rnn.get_initial_state(fake_inputs)
            # self.rnn(inputs=fake_inputs, initial_state=initial_state)
            # self.rnn.reset_states()
            raise NotImplementedError('tf.function requires different initial states for dnc')

        TensorSpecs = [([None, *state_shape], state_dtype, 'state'),
                        ([lstm_units], tf.float32, 'h'),
                        ([lstm_units], tf.float32, 'c')]
        self.rnn_states = build(self._rnn_states, TensorSpecs, sequential=False, batch_size=self.batch_size)

    @tf.function(experimental_relax_shapes=True)
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
            states: the hidden state of lstm
        """
        pwc(f'{self.name} "step" is retracing: x={x.shape}', color='cyan')
        # expand time dimension assuming x has shape `[batch_size, *state_shape]`
        x = tf.expand_dims(x, 1)
        x, states = self._common_layers(x, initial_state)
        x = tf.squeeze(x, 1)

        actor_output = self.actor(x)
        value = self.critic(x)
        assert len(actor_output.shape) == 2
        assert value.shape.ndims == 2

        if self._is_action_discrete:
            action_distribution = self.ActionDistributionType(actor_output)

            action = action_distribution.sample(one_hot=False)
            logpi = action_distribution.logp(action)
            assert action.shape.ndims == 1
            assert logpi.shape.ndims == 2
        else:
            action_distribution = self.ActionDistributionType(actor_output, self.logstd)

            raw_action = action_distribution.sample()
            logpi = action_distribution.logp(raw_action)

            # squash action
            action = tf.tanh(raw_action)
            logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)
            assert action.shape.ndims == 2
            assert logpi.shape.ndims == 2

        return action, logpi, value, states

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def det_action(self, x, initial_state):
        """ Get the deterministic actions given state x 
        
        Args:
            x: a batch of states of shape `[batch_size, *state_shape]
            initial_state: initial state for LSTM
        Returns:
            action: determinitistic action of shape `[batch_size(, action_dim)]`
            states: the hidden state of lstm
        """
        pwc(f'{self.name} "det_action" is retracing: x={x.shape}', color='cyan')
        with tf.name_scope('det_action'):
            x = tf.expand_dims(x, 1)
            x, states = self._common_layers(x, initial_state)
            x = tf.squeeze(x, 1)

            actor_output = self.actor(x)
            assert len(actor_output.shape) == 2

            if self._is_action_discrete:
                return tf.argmax(actor_output, -1), states
            else:
                return tf.tanh(actor_output), states
    
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
        pwc(f'{self.name} "train_step" is retracing: x={x.shape}, a={a.shape}', color='cyan')
        with tf.name_scope('train_step'):
            x, states = self._common_layers(x, initial_state)

            actor_output = self.actor(x)
            value = self.critic(x)
            if self._is_action_discrete:
                action_distribution = self.ActionDistributionType(actor_output)
                logpi = action_distribution.logp(a)
            else:
                action_distribution = self.ActionDistributionType(actor_output, self.logstd)
                # correction for squashed action
                # clip_but_pass_gradient is used to avoid case when a == -1, 1
                raw_action = tf.math.atanh(clip_but_pass_gradient(a, -1+1e-7, 1-1e-7))
                logpi = action_distribution.logp(tf.stop_gradient(raw_action))
                logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)

            entropy = action_distribution.entropy()

            return logpi, entropy, value, getattr(self, 'logstd', None)

    def _common_layers(self, x, initial_state):
        if hasattr(self, 'cnn'):
            x = tf.cast(x, tf.float32)
            x = x / 255.
            x = self.cnn(x)
        if hasattr(self, 'shared_mlp'):
            x = self.shared_mlp(x)

        x, h, c = self.rnn(x, initial_state=initial_state)
        
        return x, [h, c]

    def reset_states(self, states=None):
        self.rnn.reset_states(states)

    def get_initial_state(self):
        """ Get the initial states of rnn, 
        should only be called after the model is built """
        fake_inputs = tf.zeros([self.batch_size, 1, self.rnn_input_size])
        return self.rnn.get_initial_state(fake_inputs)


def create_model(model_config, state_shape, state_dtype, action_dim, is_action_discrete, n_envs):
    ac = PPOAC(
        model_config, 
        state_shape, 
        state_dtype,
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

    batch_size = np.random.randint(1, 10)
    seq_len = np.random.randint(1, 10)
    state_shape = [5]
    action_dim = np.random.randint(1, 10)
    for is_action_discrete in [True, False]:
        action_dtype = np.int32 if is_action_discrete else np.float32
        
        ac = PPOAC(config, state_shape, np.float32, action_dim, is_action_discrete, batch_size, 'ac')

        from utility.display import display_var_info

        display_var_info(ac.trainable_variables)

        # test rnn state
        x = np.random.rand(batch_size, seq_len, *state_shape)
        states = ac.get_initial_state()
        np.testing.assert_allclose(states, 0.)
        for i in range(seq_len):
            _, _, _, states = ac.step(tf.convert_to_tensor(x[:, i], tf.float32), states)
        step_states = states
        states = ac.get_initial_state()
        np.testing.assert_allclose(states, 0.)

        if is_action_discrete:
            a = np.random.randint(low=0, high=action_dim, size=(batch_size, seq_len))
        else:
            a = np.random.rand(batch_size, seq_len, action_dim)
        _, states = ac._common_layers(tf.convert_to_tensor(x, tf.float32), states)
        train_step_states = states
        np.testing.assert_allclose(step_states, train_step_states, atol=1e-5, rtol=1e-5)

