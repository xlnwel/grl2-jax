import numpy as np
from jax import lax, random, tree_util
import jax.numpy as jnp

from jax_tools import jax_assert


def static_scan(f, inputs, start):
    """
        One function which can help conduct trajectory roll-out or performing rnn
        It can be seem as a general version or an extension of hk.static_unroll
    """
    last = start
    if isinstance(start, tuple) and hasattr(start, "_asdict"):
        _keys = start._asdict().keys()
        outputs = {
            _key: [start._asdict()[_key]] for _key in _keys
        }
    elif isinstance(start, tuple):
        assert isinstance(start[0], tuple), start[0].__class__
        _keys = start[0]._asdict().keys()
        outputs = [{
            _key: [start[i]._asdict()[_key]] for _key in _keys
        } for i in range(len(start))]
    else:
        assert 0, start.__class__

    assert isinstance(inputs, tuple) or isinstance(inputs, jnp.ndarray), inputs.__class__
    indices = range(inputs.shape[0]) if isinstance(inputs, jnp.ndarray) else range(inputs[-1].shape[0])
    for index in indices:
        inp = inputs[index] if isinstance(inputs, jnp.ndarray) else tuple([inputs_item[index] if inputs_item is not None else None for inputs_item in inputs])
        last = f(last, inp)
        if isinstance(last, tuple) and hasattr(last, "_asdict"):
            for _key in outputs:
                outputs[_key].append(eval(f"last.{_key}"))
        else:
            for i in range(len(last)):
                for _key in outputs[i]:
                    outputs[i][_key].append(eval(f"last[{i}].{_key}"))
    if isinstance(outputs, dict):
        for _key in outputs:
            outputs[_key] = jnp.stack(outputs[_key], 0)
    elif isinstance(outputs, list):
        for i in range(len(outputs)):
            for _key in outputs[i]:
                outputs[i][_key] = jnp.stack(outputs[i][_key], 0)
    else:
        assert 0, outputs.__class__
    return outputs

def tree_map(f, x, *rest, is_leaf=None):
    x = tree_util.tree_map(
        lambda x, *rest: x if x is None else f(x, *rest), x, *rest, is_leaf=is_leaf)
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

def split_rng(rng, n):
    if rng is None:
        rngs = [None for _ in range(n)]
    else:
        rngs = random.split(rng, n)
    return rngs
