import numpy as np
import tensorflow as tf

from utility.display import assert_colorize, pwc


def upsample(x):
    h, w = x.get_shape().as_list()[1:-1]
    x = tf.image.resize_nearest_neighbor(x, [2 * h, 2 * w])
    return x

def safe_norm(m, axis=None, keepdims=None, epsilon=1e-6):
    """The gradient-safe version of tf.norm(...)
    it avoid nan gradient when m consists of zeros
    """
    squared_norms = tf.reduce_sum(m * m, axis=axis, keepdims=keepdims)
    return tf.sqrt(squared_norms + epsilon)

def standard_normalization(x):
    with tf.name_scope('Normalization'):
        n_dims = len(x.shape.as_list())
        mean, var = tf.nn.moments(x, list(range(n_dims-1)), keep_dims=True)
        std = tf.sqrt(var)

        x = (x - mean) / std
    
    return x

def logsumexp(value, axis=None, keepdims=False):
    if axis is not None:
        max_value = tf.reduce_max(value, axis=axis, keepdims=True)
        value0 = value - max_value    # for numerical stability
        if keepdims is False:
            max_value = tf.squeeze(max_value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value0),
                                                axis=axis, keepdims=keepdims))
    else:
        max_value = tf.reduce_max(value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value - max_value)))

def square_sum(x):
    return 2 * tf.nn.l2_loss(x)

def padding(x, kernel_size, strides, mode='constant', name=None):
    """ This function pads x so that a convolution with the same args downsamples x by a factor of strides.
    It achieves it using the following equation:
    W // S = (W - k_w + 2P) / S + 1
    """
    assert_colorize(mode.lower() == 'constant' or mode.lower() == 'reflect' or mode.lower() == 'symmetric', 
        f'Padding should be "constant", "reflect", or "symmetric", but got {mode}.')
    H, W = x.shape.as_list()[1:3]
    if isinstance(kernel_size, list) and len(kernel_size) == 2:
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size
    p_h1 = int(((H / strides - 1) * strides - H + k_h) // strides)
    p_h2 = int(((H / strides - 1) * strides - H + k_h) - p_h1)
    p_w1 = int(((W / strides - 1) * strides - W + k_w) // strides)
    p_w2 = int(((W / strides - 1) * strides - W + k_w) -p_w1)
    return tf.pad(x, [[0, 0], [p_h1, p_h2], [p_w1, p_w2], [0, 0]], mode, name=name)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])    # [N, M]

    # [1, M]
    u_var = tf.get_variable('u', [1, w_shape[-1]], 
                            initializer=tf.truncated_normal_initializer(), 
                            trainable=False)
    u = u_var
    # power iteration
    for _ in range(iteration):
        v = tf.nn.l2_normalize(tf.matmul(u, w, transpose_b=True))           # [1, N]
        u = tf.nn.l2_normalize(tf.matmul(v, w))                             # [1, M]

    sigma = tf.squeeze(tf.matmul(tf.matmul(v, w), u, transpose_b=True))     # scalar
    w = w / sigma

    with tf.control_dependencies([u_var.assign(u)]):                        # we reuse the value of u
        w = tf.reshape(w, w_shape)

    return w

def positional_encoding(indices, max_idx, dim, name='positional_encoding'):
    with tf.name_scope(name):
        # exp(-2i / d_model * log(10000))
        vals = np.array([pos * np.exp(- np.arange(0, dim, 2) / dim * np.log(10000)) for pos in range(max_idx)])
        
        params = np.zeros((max_idx, dim))
        params[:, 0::2] = np.sin(vals)    # 2i
        params[:, 1::2] = np.cos(vals)    # 2i + 1
        params = tf.convert_to_tensor(params, tf.float32)

        v = tf.nn.embedding_lookup(params, indices)

    return v

def static_scan(fn, start_state, inputs, reverse=False):
    """ sequentially apply fn to inputs, with starting state start_state 
    inputs are expected to be time-major, and the outputs of fn are expected
    to have the same structure as start_state """
    last = start_state
    outputs = [[] for _ in tf.nest.flatten(start_state)]
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        # extract inputs at step index
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        # distribute outputs
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, 0) for x in outputs]
    # reconstruct outputs to have the same structure as start_state
    return tf.nest.pack_sequence_as(start_state, outputs)
