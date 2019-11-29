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
state_shape = [5]
action_dim = 1
batch_size = 3
is_action_discrete = False
seq_len = 2
ac = PPOAC(config, state_shape, action_dim, is_action_discrete, batch_size, 'ac')

class TestClass:
    def test_rnn_states_with_initial_zeros(self):
        # step states
        x = np.random.rand(batch_size, seq_len, *state_shape)
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
        a = np.random.rand(batch_size, seq_len, action_dim)
        _, _, _, states = ac.train_step(tf.convert_to_tensor(x, tf.float32), 
            tf.convert_to_tensor(a, tf.float32), states)
        train_step_states = states

        np.testing.assert_allclose(step_states, train_step_states)

    def test_rnn_states_with_random_initial_states(self):
        # step states
        x = np.random.rand(batch_size, seq_len, *state_shape)
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
        a = np.random.rand(batch_size, seq_len, action_dim)
        _, _, _, states = ac.train_step(tf.convert_to_tensor(x, tf.float32), 
            tf.convert_to_tensor(a, tf.float32), states)
        train_step_states = states

        np.testing.assert_allclose(step_states, train_step_states)


