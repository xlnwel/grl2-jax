from collections import namedtuple
import tensorflow as tf
from tensorflow.keras import layers, activations, initializers, regularizers, constraints


LSTMState = namedtuple('LSTMState', ['h', 'c'])

class LSTMCell(layers.Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 use_ln=False,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.use_ln = use_ln

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.state_size = LSTMState(h=self.units, c=self.units)
        self.output_size = self.units

    def build(self, input_shapes):
        input_dim = input_shapes[0][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return tf.concat([
                      self.bias_initializer((self.units,), *args, **kwargs),
                      initializers.Ones()((self.units,), *args, **kwargs),
                      self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ], -1)
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
              shape=(self.units * 4,),
              name='bias',
              initializer=bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_ln:
            self.x_ln = layers.LayerNormalization()
            self.h_ln = layers.LayerNormalization()
            self.c_ln = layers.LayerNormalization()
        else:
            self.x_ln = lambda x: x
            self.h_ln = lambda x: x
            self.c_ln = lambda x: x

    def call(self, x, states):
        x, mask = tf.nest.flatten(x)
        h, c = states
        if mask is not None:
            h = h * mask
            c = c * mask
        
        x = self.x_ln(tf.matmul(x, self.kernel)) + self.h_ln(tf.matmul(h, self.recurrent_kernel))
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        i, f, c_, o = tf.split(x, 4, 1)
        i, f, o = self.recurrent_activation(i), self.recurrent_activation(f), self.recurrent_activation(o)
        c_ = self.activation(c_)
        c = f * c + i * c_
        h = o * self.activation(self.c_ln(c))
            
        return h, LSTMState(h, c)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        state_size = self.state_size
        if inputs is not None:
            assert batch_size is None or batch_size == tf.shape(inputs)[0]
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        return LSTMState(
            h=tf.zeros([batch_size, state_size[0]], dtype),
            c=tf.zeros([batch_size, state_size[1]], dtype))
