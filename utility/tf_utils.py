import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utility.display import assert_colorize, pwc


def configure_gpu(i=0):
    """Configure gpu for Tensorflow"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # restrict TensorFlow to only use the i-th GPU
            tf.config.experimental.set_visible_devices(gpus[i], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            pwc(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU', 
                color='cyan')
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            pwc(e)

def get_TensorSpecs(TensorSpecs, sequential=False, batch_size=None):
    """Construct a dict/list of TensorSpecs
    
    Args:
        TensorSpecs: A dict/list/tuple of arguments for tf.TensorSpec,
        sequential: A boolean, if True, batch_size must be specified, and 
            the result TensorSpec will have fixed batch_size and a time dimension
        batch_size: Specifies the batch size
    Returns: 
        If TensorSpecs is a dict, return a dict of TensorSpecs with names 
        as they are in TensorSpecs. Otherwise, return a list of TensorSpecs
    """
    if sequential:
        assert_colorize(batch_size, 
            f'For sequential data, please specify batch size for RNN states')
        default_shape = [batch_size, None]
    else:
        default_shape = [batch_size]
    if isinstance(TensorSpecs, dict):
        name = TensorSpecs.keys()
        tensorspecs = TensorSpecs.values()
    else:
        name = None
        tensorspecs = TensorSpecs
    assert_colorize(isinstance(tensorspecs, (list, tuple)), 
        'Expect tensorspecs to be a dict/list/tuple of arguments for tf.TensorSpec, '
        f'but get {TensorSpecs}\n')
    tensorspecs = [tf.TensorSpec(shape=default_shape+list(s) if s else s, dtype=d, name=n)
         for s, d, n in tensorspecs]
    if name:
        return dict(zip(name, tensorspecs))
    else:
        return tensorspecs

def build(func, TensorSpecs, sequential=False, batch_size=None):
    """Build a concrete function of func

    Args:
        func: A function decorated by @tf.function
        sequential: A boolean, if True, batch_size must be specified, and 
            the result TensorSpec will have fixed batch_size and a time dimension
        batch_size: Specifies the batch size
    Returns:
        A concrete function of func
    """
    TensorSpecs = get_TensorSpecs(TensorSpecs, sequential, batch_size)

    if isinstance(TensorSpecs, dict):
        return func.get_concrete_function(**TensorSpecs)
    else: 
        return func.get_concrete_function(*TensorSpecs)
    

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
    

def n_step_target(reward, done, nth_value, gamma, steps):
    with tf.name_scope('n_step_target'):
        n_step_target = tf.stop_gradient(reward 
                                        + gamma**steps
                                        * (1 - done)
                                        * nth_value, name='n_step_target')

    return n_step_target


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
