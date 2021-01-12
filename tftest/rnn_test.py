import numpy as np
import tensorflow as tf

from nn.rnn import LSTM


class LSTMTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()

        self.config = dict(
            units=16,
            use_ln=False
        )

        self.bs = 32
        self.seqlen = 16
        self.x_dim = 5

    def _get_lstm(self, config=None):
        config = config or self.config
        lstm = LSTM(config)

        return lstm
    
    def test_state_with_zero_states(self):
        for ln in [True, False]:
            config = self.config.copy()
            config['use_ln'] = ln
            lstm = self._get_lstm(config)

            mask = np.random.randint(0, 2, (self.bs, self.seqlen)).astype(np.float32)
            x = np.random.rand(self.bs, self.seqlen, self.x_dim).astype(np.float32)
            # step states
            states = lstm.get_initial_state(x)
            for s in states:
                self.assertAllClose(s, tf.zeros_like(s))
            for i in range(self.seqlen):
                _, states = lstm(
                    tf.convert_to_tensor(x[:, i:i+1], tf.float32), 
                    states, mask[:, i:i+1])
            step_states = states
            # second round
            states = lstm.get_initial_state(batch_size=self.bs)
            for s in states:
                self.assertAllClose(s, tf.zeros_like(s))
            for i in range(self.seqlen):
                _, states = lstm(
                    tf.convert_to_tensor(x[:, i:i+1], tf.float32), 
                    states, mask[:, i:i+1])
            step_states2 = states
            self.assertAllClose(step_states, step_states2)
            
            # rnn_state states
            states = lstm.get_initial_state(batch_size=self.bs)
            for s in states:
                self.assertAllClose(s, tf.zeros_like(s))
            _, states = lstm(tf.convert_to_tensor(x, tf.float32), 
                states, mask)
            rnn_state = states
            self.assertAllClose(step_states, rnn_state)
    
    def test_state_with_random_states(self):
        for ln in [True, False]:
            config = self.config.copy()
            config['use_ln'] = ln
            lstm = self._get_lstm(config)

            mask = np.random.randint(0, 2, (self.bs, self.seqlen)).astype(np.float32)
            x = np.random.rand(self.bs, self.seqlen, self.x_dim).astype(np.float32)
            rnn_state_shape = (self.bs, self.config['units'])
            init_state = [tf.random.normal((rnn_state_shape)), 
                    tf.random.normal(rnn_state_shape)]
            # step states
            states = init_state
            for i in range(self.seqlen):
                _, new_states = lstm(
                    tf.convert_to_tensor(x[:, i:i+1], tf.float32), 
                    states, mask[:, i:i+1])
                self.assertNotAllClose(tuple(states), tuple(new_states))
                states = new_states
            step_states = states
            # second round
            states = init_state
            for i in range(self.seqlen):
                _, new_states = lstm(
                    tf.convert_to_tensor(x[:, i:i+1], tf.float32), 
                    states, mask[:, i:i+1])
                self.assertNotAllClose(tuple(states), tuple(new_states))
                states = new_states
            step_states2 = states
            self.assertAllClose(step_states, step_states2)
            
            # rnn_state states
            states = init_state
            _, rnn_state = lstm(tf.convert_to_tensor(x, tf.float32), 
                states, mask)
            self.assertAllClose(step_states, rnn_state)
            self.assertNotAllClose(tuple(states), tuple(rnn_state))
    
    def test_mask(self):
        lstm = self._get_lstm()
        mask1 = np.random.randint(0, 2, (self.bs, self.seqlen)).astype(np.float32)
        mask2 = np.random.randint(0, 2, (self.bs, self.seqlen)).astype(np.float32)
        while np.all(mask1 == mask2):
            mask2 = np.random.randint(0, 2, (self.bs, self.seqlen)).astype(np.float32)
        x = np.random.rand(self.bs, self.seqlen, self.x_dim).astype(np.float32)    
        rnn_state_shape = (self.bs, self.config['units'])
        init_state = [tf.random.normal((rnn_state_shape)), 
                    tf.random.normal(rnn_state_shape)]
        _, state1 = lstm(x, init_state, mask1)
        _, state1_2 = lstm(x, init_state, mask1)
        _, state2 = lstm(x, init_state, mask2)
    
        self.assertAllClose(state1, state1_2)
        self.assertNotAllClose(state1, state2)

    def test_mask_grads(self):
        lstm = self._get_lstm()
        mask1 = np.random.randint(0, 2, (self.bs, self.seqlen)).astype(np.float32)
        mask2 = mask1[:, -1:] = 0
        x1 = np.random.rand(self.bs, self.seqlen, self.x_dim).astype(np.float32)
        x2 = x1[:, -1:]
        state = lstm.get_initial_state(x1)
        with tf.GradientTape() as t:
            y1, state1 = lstm(x1, state, mask1)
            y1 = y1[:, -1:]
            loss1 = tf.reduce_mean(.5 * (1 - y1)**2)
        g1 = t.gradient(loss1, lstm.variables)

        with tf.GradientTape() as t:
            _, state2 = lstm(x1[:, :-1], state, mask1[:, :-1])
            state2 = [tf.stop_gradient(s) for s in state2]
            y2, state2 = lstm(x1[:, -1:], state2, mask1[:, -1:])
            loss2 = tf.reduce_mean(.5 * (1 - y2)**2)
        g2 = t.gradient(loss2, lstm.variables)

        self.assertAllClose(y1, y2)
        self.assertAllClose(state1, state2)
        self.assertAllClose(loss1, loss2)
        self.assertAllClose(g1, g2)