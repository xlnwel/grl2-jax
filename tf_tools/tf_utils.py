from typing import List
import numpy as np
import tensorflow as tf


def tensor2numpy(x):
    return tf.nest.map_structure(
        lambda x: x.numpy() if isinstance(x, tf.Tensor) else x, x)

def numpy2tensor(x):
    return tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x) if x is not None else x, x)

def gather(data, i, axis=0):
    data = tf.nest.map_structure(
        lambda x: x if x is None else tf.gather(x, i, axis=axis), data)
    return data

def safe_ratio(pi, mu, eps=1e-8):
    return pi / (mu + eps)

def safe_norm(m, axis=None, keepdims=None, epsilon=1e-6):
    """The gradient-safe version of tf.norm(...)
    it avoid nan gradient when m consists of zeros
    """
    squared_norms = tf.reduce_sum(m * m, axis=axis, keepdims=keepdims)
    return tf.sqrt(squared_norms + epsilon)

def reduce_mean(x, mask=None, n=None, axis=None):
    if mask is not None and n is None:
        n = tf.reduce_sum(mask)
        n = tf.where(n == 0, 1., n)
    return tf.reduce_mean(x, axis=axis) \
        if mask is None else tf.reduce_sum(x * mask, axis=axis) / n

def reduce_moments(x, mask=None, n=None, axis=None):
    if mask is not None and n is None:
        n = tf.reduce_sum(mask)
        n = tf.where(n == 0, 1., n)
    mean = reduce_mean(x, mask=mask, n=n, axis=axis)
    var = reduce_mean((x - mean)**2, mask=mask, n=n, axis=axis)
    return mean, var

def standard_normalization(
    x, 
    zero_center=True, 
    mask=None, 
    n=None, 
    axis=None, 
    epsilon=1e-8, 
    clip=None
):
    mean, var = reduce_moments(x, mask=mask, n=n, axis=axis)
    std = tf.sqrt(var + epsilon)
    if zero_center:
        x = x - mean
    x = x / std
    if clip is not None:
        x = tf.clip_by_value(x, -clip, clip)

    return x

def explained_variance(y, pred, axis=None, mask=None):
    if None in y.shape:
        assert y.shape.ndims == pred.shape.ndims, (y.shape, pred.shape)
    else:
        assert y.shape == pred.shape, (y.shape, pred.shape)
    y_var = tf.math.reduce_variance(y, axis=axis)
    diff_var = tf.math.reduce_variance(y - pred, axis=axis)
    ev = tf.maximum(-1., 1-(diff_var / y_var))
    ev = reduce_mean(ev, mask=mask)

    return ev

def softmax(x, tau, axis=-1):
    """ sfotmax(x / tau) """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max
    return tf.nn.softmax(x / tau, axis=axis)

def logsumexp(x, tau, axis=None, keepdims=False):
    """ reimplementation of tau * tf.logsumexp(x / tau), it turns out 
    that tf.logsumexp is numerical stable """
    x /= tau
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max    # for numerical stability
    if keepdims is False:
        x_max = tf.squeeze(x_max)
    y = x_max + tf.math.log(tf.reduce_sum(
        tf.exp(x), axis=axis, keepdims=keepdims))
    return tau * y

def log_softmax(x, tau, axis=-1):
    """ tau * log_softmax(x / tau) """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max
    x = x - tau * tf.reduce_logsumexp(x / tau, axis=axis, keepdims=True)
    return x

def square_sum(x):
    return 2 * tf.nn.l2_loss(x)

def padding(x, kernel_size, strides, mode='constant', name=None):
    """ This function pads x so that a convolution with the same args downsamples x by a factor of strides.
    It achieves it using the following equation:
    W // S = (W - k_w + 2P) / S + 1
    """
    assert mode.lower() == 'constant' or mode.lower() == 'reflect' or mode.lower() == 'symmetric', \
        f'Padding should be "constant", "reflect", or "symmetric", but got {mode}.'
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

def spectral_norm(w, u_var, iterations=1):
    w_shape = w.shape
    if len(w_shape) != 2:
        w = tf.reshape(w, [-1, w_shape[-1]])    # [N, M]

    u = u_var
    assert u.shape == [1, w_shape[-1]]
    # power iteration
    for i in range(iterations):
        v = tf.nn.l2_normalize(tf.matmul(u, w, transpose_b=True))           # [1, N]
        u = tf.nn.l2_normalize(tf.matmul(v, w))                             # [1, M]

    sigma = tf.squeeze(tf.matmul(tf.matmul(v, w), u, transpose_b=True))     # scalar
    w = w / sigma

    u_var.assign(u)
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

def static_scan(fn, start, inputs, reverse=False):
    """ Sequentially apply fn to inputs, with starting state start.
    inputs are expected to be time-major, and the outputs of fn are expected
    to have the same structure as start. 
    This function is equivalent to 
    tf.scan(
        fn=fn
        elems=inputs, 
        initializer=start,
        parallel_iterations=1,
        reverse=reverse
    )
    In practice, we find it's faster than tf.scan
    """
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(tf.nest.flatten(inputs)[0].shape[0])
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
    outputs = [tf.stack(x) for x in outputs]
    # reconstruct outputs to have the same structure as start
    return tf.nest.pack_sequence_as(start, outputs)

def get_stoch_state(x, min_std):
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + min_std
    stoch = mean + tf.random.normal(tf.shape(mean)) * std
    return mean, std, stoch

def assert_rank(tensors, rank=None):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    rank = rank or tensors[0].shape.ndims
    for tensor in tensors:
        if tensor is None:
            continue
        tensor_shape = tf.TensorShape(tensor.shape)
        tensor_shape.assert_has_rank(rank)

def assert_shape_compatibility(tensors):
    assert isinstance(tensors, (list, tuple)), tensors
    union_of_shapes = tf.TensorShape(None)
    for tensor in tensors:
        if tensor is None:
            continue
        tensor_shape = tf.TensorShape(tensor.shape)
        union_of_shapes = union_of_shapes.merge_with(tensor_shape)

def assert_rank_and_shape_compatibility(tensors: List[tf.Tensor], rank=None):
    """Asserts that the tensors have the correct rank and compatible shapes.

    Shapes (of equal rank) are compatible if corresponding dimensions are all
    equal or unspecified. E.g. `[2, 3]` is compatible with all of `[2, 3]`,
    `[None, 3]`, `[2, None]` and `[None, None]`.

    Args:
        tensors: List of tensors.
        rank: A scalar specifying the rank that the tensors passed need to have.

    Raises:
        ValueError: If the list of tensors is empty or fail the rank and mutual
        compatibility asserts.
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    rank = rank or tensors[0].shape.ndims
    union_of_shapes = tf.TensorShape(None)
    for tensor in tensors:
        if tensor is None:
            continue
        tensor_shape = tf.TensorShape(tensor.shape)
        tensor_shape.assert_has_rank(rank)
        union_of_shapes = union_of_shapes.merge_with(tensor_shape)

def split_data(x, next_x=None, axis=1):
    if isinstance(x, (list, tuple)):
        if next_x is None:
            next_x = [None for _ in x]
        x, next_x = list(zip(*[split_data(xx, next_xx, axis=axis) 
            for xx, next_xx in zip(x, next_x)]))
        return x, next_x
    if x is None:
        return None, None
    if next_x is None:
        n = x.shape[axis]
        _, next_x = tf.split(x, [1, n-1], axis=axis)
        x, _ = tf.split(x, [n-1, 1], axis=axis)

    return x, next_x

def assert_finite(vars, prefix="", name=None):
    if isinstance(vars, (list, tuple)):
        for v in vars:
            assert_finite(v, prefix)
    elif isinstance(vars, dict):
        for k, v in vars.items():
            assert_finite(v, prefix, k)
    elif vars is not None:
        if name is None:
            name = vars.name
        tf.debugging.assert_all_finite(vars, f'{prefix}: Bad {name}')
