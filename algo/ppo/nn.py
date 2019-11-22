import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utility.display import pwc
from utility.tf_utils import build
from utility.rl_utils import clip_but_pass_gradient, logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.layers.func import mlp_layers, dnc_rnn
from nn.initializers import get_initializer


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
        shared_mlp_units = config['shared_mlp_units']
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
        initializer_name = config.get('kernel_initializer', 'he_uniform')
        kernel_initializer = get_initializer(initializer_name)

        self.batch_size = batch_size

        """ Network definition """
        # shared mlp layers
        self.shared_mlp = mlp_layers(shared_mlp_units, 
                                    norm=norm, 
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
            # this has to be done for DNC, for some obscure reason, we have to pass
            # in initial_state for the first time so that the RNN can know a 
            # high-order state_size
            tf.summary.experimental.set_step(0)
            input_size = shared_mlp_units[-1]
            fake_inputs = tf.zeros([batch_size, 1, input_size])
            initial_state = self.rnn.get_initial_state(fake_inputs)
            self.rnn(inputs=fake_inputs, initial_state=initial_state)
            self.rnn.reset_states()

        # TensorSpecs = [(state_shape, tf.float32, 'state')]
        # self.step = build(self._step, TensorSpecs, sequential=False, batch_size=batch_size)

    @tf.function
    @tf.Module.with_name_scope
    def step(self, x):
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
        pwc(f'{self.name} "step" is retracing: x={x}', color='cyan')
        # expand time dimension assuming x has shape `[batch_size, *state_shape]`
        x = tf.expand_dims(x, 1)
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
    @tf.Module.with_name_scope
    def det_action(self, x):
        """ Get the deterministic actions given state x 
        
        Args:
            x: a batch of states of shape `[batch_size, *state_shape]
        Returns:
            determinitistic action of shape `[batch_size, action_dim]`
        """
        pwc(f'{self.name} "det_action" is retracing: x={x}', color='cyan')
        with tf.name_scope('det_action'):
            x = tf.expand_dims(x, 1)
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
        pwc(f'{self.name} "train_step" is retracing: x={x}, a={a}', color='cyan')
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
        pwc(f'{self.name} "common_layer" is retracing', color='cyan')
        
        for l in self.shared_mlp:
            x = l(x)

        x = self.rnn(x)
        
        return x

    def _head(self, x, layers):
        for l in layers:
            x = l(x)
        
        return x

    def reset_states(self, states=None):
        self.rnn.reset_states(states)


def create_model(model_config, state_shape, action_dim, is_action_discrete, n_envs):
    ac = PPOAC(model_config, 
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
        kernel_initializer='he'
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
    pwc(ac.rnn.states)
    for i in range(seq_len):
        y = ac.step(tf.convert_to_tensor(x[:, i], tf.float32))
    pwc(ac.rnn.states)
    ac.reset_states()
    pwc(ac.rnn.states)
    a = np.random.rand(batch_size, seq_len, action_dim)
    ac.train_step(tf.convert_to_tensor(x, tf.float32), tf.convert_to_tensor(a, tf.float32))
    pwc(ac.rnn.states)
