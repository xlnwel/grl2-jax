import numpy as np
from jax import lax, random, tree_util
import jax.numpy as jnp

from jax_tools import jax_assert


def tree_map(f, x):
    x = tree_util.tree_map(
        lambda x: x if x is None else f(x), x)
    return x

def random_generator(seed):
    key = random.PRNGKey(seed)
    return key

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
        _, next_x = jnp.split(x, [1], axis=axis)
        x, _ = jnp.split(x, [n-1], axis=axis)

    return x, next_x

def time_major(*args, axis):
    jax_assert.assert_shape_compatibility(args)
    dims = list(range(args[0].ndim))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if len(args) == 1:
        args = args[0]
    if axis != 0:
        args = tree_map(lambda x: lax.transpose(x, dims), args)
    return dims, args

def undo_time_major(*args, dims, axis):
    jax_assert.assert_shape_compatibility(args)
    if len(args) == 1:
        args = args[0]
    if axis != 0:
        args = tree_map(lambda x: lax.transpose(x, dims), args)
    return args

def jnp2np(data):
    data = tree_map(lambda x: np.array(x) 
        if isinstance(x, jnp.DeviceArray) else x, data)
    return data

def compute_norms(tree):
    tree = tree_util.tree_map(jnp.linalg.norm, tree)
    return tree
