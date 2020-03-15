import numpy as np
import tensorflow as tf

from algo.ppo2.nn import PPOAC

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
obs_shape = [5]
action_dim = np.random.randint(1, 10)
batch_size = np.random.randint(1, 10)
seq_len = np.random.randint(1, 10)

class TestClass:
    def test_rnn_states_with_initial_zeros(self):
        for is_action_discrete in [True, False]:
            ac = PPOAC(config, obs_shape, np.float32, action_dim, is_action_discrete, batch_size, 'ac')

            # step states
            x = np.random.rand(batch_size, seq_len, *obs_shape)
            states = ac.get_initial_state()
            np.testing.assert_allclose(states, 0)
            for i in range(seq_len):
                _, _, _, states = ac.step(tf.convert_to_tensor(x[:, i], tf.float32), states)
            step_states = states
            # second round
            states = ac.get_initial_state()
            for i in range(seq_len):
                _, _, _, states = ac.step(tf.convert_to_tensor(x[:, i], tf.float32), states)
            step_states2 = states
            np.testing.assert_allclose(step_states, step_states2)

            # det_action states
            states = ac.get_initial_state()
            np.testing.assert_allclose(states, 0)
            for i in range(seq_len):
                _, states = ac.det_action(tf.convert_to_tensor(x[:, i], tf.float32), states)
            det_action_states = states
            np.testing.assert_allclose(step_states, det_action_states)
            
            # rnn_states states
            states = ac.get_initial_state()
            np.testing.assert_allclose(states, 0)
            states = ac.rnn_states(tf.convert_to_tensor(x, tf.float32), *states)
            rnn_states = states
            np.testing.assert_allclose(step_states, rnn_states)

            # train_step states
            states = ac.get_initial_state()
            np.testing.assert_allclose(states, 0)
            if is_action_discrete:
                a = np.random.randint(low=0, high=action_dim, size=(batch_size, seq_len))
            else:
                a = np.random.rand(batch_size, seq_len, action_dim)
            action_dtype = np.int32 if is_action_discrete else np.float32
            _, states = ac._common_layers(tf.convert_to_tensor(x, tf.float32), states)
            train_step_states = states

            np.testing.assert_allclose(step_states, train_step_states)

    def test_rnn_states_with_random_initial_states(self):
        for is_action_discrete in [True, False]:
            ac = PPOAC(config, obs_shape, np.float32, action_dim, is_action_discrete, batch_size, 'ac')

            # step states
            x = np.random.rand(batch_size, seq_len, *obs_shape)
            states = ac.get_initial_state()
            rnn_state_shape = np.shape(states[0])
            initial_state = [tf.random.normal(rnn_state_shape), 
                tf.random.normal(rnn_state_shape)]
            states = initial_state
            for i in range(seq_len):
                _, _, _, states = ac.step(tf.convert_to_tensor(x[:, i], tf.float32), states)
            step_states = states
            # second round
            states = initial_state
            for i in range(seq_len):
                _, _, _, states = ac.step(tf.convert_to_tensor(x[:, i], tf.float32), states)
            step_states2 = states
            np.testing.assert_allclose(step_states, step_states2)

            # det_action states
            states = initial_state
            for i in range(seq_len):
                _, states = ac.det_action(tf.convert_to_tensor(x[:, i], tf.float32), states)
            det_action_states = states
            np.testing.assert_allclose(step_states, det_action_states)
            
            # rnn_states states
            states = initial_state
            states = ac.rnn_states(tf.convert_to_tensor(x, tf.float32), *states)
            rnn_states = states
            np.testing.assert_allclose(step_states, rnn_states)

            # train_step states
            states = initial_state
            if is_action_discrete:
                a = np.random.randint(low=0, high=action_dim, size=(batch_size, seq_len))
            else:
                a = np.random.rand(batch_size, seq_len, action_dim)
            action_dtype = np.int32 if is_action_discrete else np.float32
            _, states = ac._common_layers(tf.convert_to_tensor(x, tf.float32), states)
            train_step_states = states

            np.testing.assert_allclose(step_states, train_step_states)


