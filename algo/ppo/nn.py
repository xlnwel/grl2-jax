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
            rnn_config=dict(return_sequences=True, stateful=True)
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

        # RNN layer
        self.rnn_input_size = self.cnn.out_size if cnn_name else shared_mlp_units[-1]
        if use_dnc:
            self.rnn = dnc_rnn(**dnc_config)
        else:
            self.rnn = layers.LSTM(lstm_units, return_sequences=True, stateful=True, 
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
            # this has to be done for DNC, for some obscure reason, we have to pass
            # in initial_state for the first time so that the RNN can know a 
            # high-order state_size
            input_size = shared_mlp_units[-1]
            fake_inputs = tf.zeros([batch_size, 1, input_size])
            initial_state = self.rnn.get_initial_state(fake_inputs)
            self.rnn(inputs=fake_inputs, initial_state=initial_state)
            self.rnn.reset_states()

        TensorSpecs = [(state_shape, tf.float32, 'state')]
        self.step = build(self._step, TensorSpecs, sequential=False, batch_size=batch_size)

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def _step(self, x):
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
        pwc(f'{self.name} "step" is retracing: x={x.shape}', color='cyan')
        # expand time dimension assuming x has shape `[batch_size, *state_shape]`
        x = tf.expand_dims(x, 1)
        x = self._common_layers(x)
        x = tf.squeeze(x, 1)

        actor_output = self.actor(x)
        value = self.critic(x)
        assert len(actor_output.shape) == 2
        assert len(value.shape) == 2

        if self._is_action_discrete:
            action_distribution = self.ActionDistributionType(actor_output)

            action = action_distribution.sample(one_hot=False)
            logpi = action_distribution.logp(action)
            assert len(action.shape) == 1
            assert len(logpi.shape) == 2
        else:
            action_distribution = self.ActionDistributionType(actor_output, self.logstd)

            raw_action = action_distribution.sample()
            logpi = action_distribution.logp(raw_action)

            # squash action
            action = tf.tanh(raw_action)
            logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)
            assert len(action.shape) == 2
            assert len(logpi.shape) == 2
        
        return action, logpi, value

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def det_action(self, x):
        """ Get the deterministic actions given state x 
        
        Args:
            x: a batch of states of shape `[batch_size, *state_shape]
        Returns:
            action: determinitistic action of shape `[batch_size, action_dim]`
        """
        pwc(f'{self.name} "det_action" is retracing: x={x.shape}', color='cyan')
        with tf.name_scope('det_action'):
            x = tf.expand_dims(x, 1)
            x = self._common_layers(x)
            x = tf.squeeze(x, 1)

            actor_output = self.actor(x)
            assert len(actor_output.shape) == 2

            if self._is_action_discrete:
                return tf.argmax(actor_output, -1)
            else:
                return tf.tanh(actor_output)
    
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
        pwc(f'{self.name} "train_step" is retracing: x={x.shape}, a={a.shape}', color='cyan')
        with tf.name_scope('train_step'):
            x = self._common_layers(x)

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

    def _common_layers(self, x):
        if hasattr(self, 'cnn'):
            x = tf.cast(x, tf.float32)
            x = x / 255.
            x = self.cnn(x)
        if hasattr(self, 'shared_mlp'):
            x = self.shared_mlp(x)

        x = self.rnn(x)
        
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

    batch_size = np.random.randint(1, 10)
    seq_len = np.random.randint(1, 10)
    state_shape = [5]
    action_dim = np.random.randint(1, 10)
    for is_action_discrete in [True, False]:
        action_dtype = np.int32 if is_action_discrete else np.float32
        
        ac = PPOAC(config, state_shape, action_dim, is_action_discrete, batch_size, 'ac')

        from utility.display import display_var_info

        display_var_info(ac.trainable_variables)

        # test rnn state
        x = np.random.rand(batch_size, seq_len, *state_shape)
        
        states = [s.numpy() for s in ac.rnn.states]
        np.testing.assert_allclose(states, 0.)
        for i in range(seq_len):
            y = ac.step(tf.convert_to_tensor(x[:, i], tf.float32))
        step_states = [s.numpy() for s in ac.rnn.states]
        ac.reset_states()
        states = [s.numpy() for s in ac.rnn.states]
        np.testing.assert_allclose(states, 0.)
        if is_action_discrete:
            a = np.random.randint(low=0, high=action_dim, size=(batch_size, seq_len))
        else:
            a = np.random.rand(batch_size, seq_len, action_dim)
        ac.train_step(tf.convert_to_tensor(x, tf.float32), tf.convert_to_tensor(a, action_dtype))
        train_step_states = [s.numpy() for s in ac.rnn.states]
        np.testing.assert_allclose(step_states, train_step_states)

