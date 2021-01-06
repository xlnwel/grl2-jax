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
            lstm = self._get_lstm()

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
            lstm = self._get_lstm()

            mask = np.random.randint(0, 2, (self.bs, self.seqlen)).astype(np.float32)
            x = np.random.rand(self.bs, self.seqlen, self.x_dim).astype(np.float32)
            rnn_state_shape = (self.bs, self.config['units'])
            init_state = [tf.random.normal((rnn_state_shape)), 
                    tf.random.normal(rnn_state_shape)]
            # step states
            states = init_state
            for i in range(self.seqlen):
                _, states = lstm(
                    tf.convert_to_tensor(x[:, i:i+1], tf.float32), 
                    states, mask[:, i:i+1])
            step_states = states
            # second round
            states = init_state
            for i in range(self.seqlen):
                _, states = lstm(
                    tf.convert_to_tensor(x[:, i:i+1], tf.float32), 
                    states, mask[:, i:i+1])
            step_states2 = states
            self.assertAllClose(step_states, step_states2)
            
            # rnn_state states
            states = init_state
            _, states = lstm(tf.convert_to_tensor(x, tf.float32), 
                states, mask)
            rnn_state = states
            self.assertAllClose(step_states, rnn_state)
