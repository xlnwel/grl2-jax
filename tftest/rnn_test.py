import numpy as np
import tensorflow as tf

from nn.rnn import LSTMCell, LSTM


class LSTMTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()

        self.config = dict(
            units=16,
            use_ln=False,
            state_mask=True
        )

        self.bs = 32
        self.seqlen = 16
        self.x_dim = 5

    def _get_lstm(self, config=None):
        config = config or self.config
        lstm = LSTM(config)

        return lstm
    
    def _get_x(self, bs=None, seqlen=None):
        bs = bs or self.bs
        seqlen = seqlen or self.seqlen
        x = np.random.rand(bs, seqlen, self.x_dim).astype(np.float32)
        return x
    
    def _get_mask(self, bs=None, seqlen=None):
        bs = bs or self.bs
        seqlen = seqlen or self.seqlen
        mask = np.random.randint(0, 2, (bs, seqlen)).astype(np.float32)
        return mask

    def test_lstm(self):
        for seed in np.random.randint(0, 100, 3):
            mask = self._get_mask()
            x = self._get_x()

            tf.random.set_seed(seed)
            c = LSTMCell(self.config['units'])
            l = tf.keras.layers.RNN(c, return_sequences=True, return_state=True)        
            y = l((x, mask[..., None]), initial_state=None)
            y, s = y[0], y[1:]

            tf.random.set_seed(seed)
            l = LSTM(self.config)
            my, ms = l(x, None, mask)
            tf.debugging.assert_near(y, my)
            tf.debugging.assert_near(s, ms)

    def test_state_with_zero_states(self):
        for ln in [True, False]:
            config = self.config.copy()
            config['use_ln'] = ln
            lstm = self._get_lstm(config)

            mask = self._get_mask()
            x = self._get_x()
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

            mask = self._get_mask()
            x = self._get_x()
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
        mask1 = self._get_mask()
        mask2 = self._get_mask()
        while np.all(mask1 == mask2):
            mask2 = self._get_mask()
        x = self._get_x()    
        rnn_state_shape = (self.bs, self.config['units'])
        init_state = [tf.random.normal((rnn_state_shape)), 
                    tf.random.normal(rnn_state_shape)]
        _, state1 = lstm(x, init_state, mask1)
        _, state1_2 = lstm(x, init_state, mask1)
        _, state2 = lstm(x, init_state, mask2)
    
        self.assertAllClose(state1, state1_2)
        self.assertNotAllClose(state1, state2)

    def test_mask_grads(self):
        def loss(y):
            return tf.reduce_mean(.5 * (1 - y)**2)
        for n in np.random.randint(1, self.seqlen/2, 3):
            lstm = self._get_lstm()
            mask1 = self._get_mask()
            mask1[:, -n:] = 0
            x1 = self._get_x()
            state = lstm.get_initial_state(x1)
            with tf.GradientTape() as t:
                y1, state1 = lstm(x1, state, mask1)
                y1 = y1[:, -n:]
                loss1 = loss(y1)
            g1 = t.gradient(loss1, lstm.variables)

            with tf.GradientTape() as t:
                _, state2 = lstm(x1[:, :-n], state, mask1[:, :-n])
                state2 = [tf.stop_gradient(s) for s in state2]
                y2, state2 = lstm(x1[:, -n:], state2, mask1[:, -n:])
                loss2 = loss(y2)
            g2 = t.gradient(loss2, lstm.variables)

            self.assertAllClose(y1, y2)
            self.assertAllClose(state1, state2)
            self.assertAllClose(loss1, loss2)
            self.assertAllClose(g1, g2)
    
    def test_state_mask(self):
        for n in np.random.randint(1, self.seqlen/2, 3):
            lstm = self._get_lstm()
            mask = np.ones((self.bs, self.seqlen))
            mask[:, -n] = 0
            x = self._get_x()
            y1, state1 = lstm(x, None, mask)
            y1 = y1[:, -n:]
            y2, state2 = lstm(x[:, -n:], None, np.ones((self.bs, n)))
        
            self.assertAllClose(state1, state2)
            self.assertAllClose(y1, y2)

    def test_no_state_mask(self):
        config = self.config    # no need to copy as any changes only visible here
        config['state_mask'] = False
        for n in np.random.randint(1, self.seqlen/2, 3):
            lstm = self._get_lstm()
            mask = np.ones((self.bs, self.seqlen))
            mask[:, -n] = 0
            x = self._get_x()
            y1, state1 = lstm(x, None, mask)
            y1 = y1[:, -n:]
            y2, state2 = lstm(x[:, -n:], None, np.ones((self.bs, n)))
            print(y1)
            print(y2)
            self.assertNotAllClose(state1, state2)
            self.assertNotAllClose(y1, y2)
