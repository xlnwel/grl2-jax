import ast
import collections
import inspect
import itertools
import math
import multiprocessing
import os
import random
import numpy as np
import tensorflow as tf

from utility.typing import AttrDict


def dict2AttrDict(config: dict):
    attr_config = AttrDict(**config)
    for k, v in attr_config.items():
        if isinstance(v, dict):
            attr_config[k] = dict2AttrDict(v)

    return attr_config

def deep_update(source: dict, target:dict):
    for k, v in target.items():
        if isinstance(v, collections.abc.Mapping):
            assert k in source, f'{k} does not exist in {source}'
            assert isinstance(source[k], collections.abc.Mapping), \
                f'Inconsistent types: {type(v)} vs {type(source[k])}'
            source[k] = deep_update(source.get(k, {}), v)
        else:
            source[k] = v
    return source

def eval_config(config):
    for k, v in config.items():
        if isinstance(v, str):
            try:
                v = float(v)
            except:
                pass
        if isinstance(v, float) and v == int(v):
            v = int(v)
        config[k] = v
    return config

def config_attr(obj, config: dict, filter_dict: bool=False):
    """ Add values in config as attributes of obj

    Args:
        obj: the target object to which we add attributes
        config: values associated to uppercase keys
            are added as public attributes, while those
            associated to lowercase keys are added as
            private attributes
        filter_dict: whether to omit dictionaries
    """
    config = dict2AttrDict(config)
    setattr(obj, 'config', config)
    for k, v in config.items():
        if filter_dict and isinstance(v, dict):
            continue
        if k.islower():
            k = f'_{k}'
        if isinstance(v, str):
            try:
                v = float(v)
            except:
                pass
        if isinstance(v, float) and v == int(v):
            v = int(v)
        setattr(obj, k, v)

def to_int(s):
    return int(float(s))
    
def to_array32(x):
    x = np.array(x, copy=False)
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    elif x.dtype == np.int64:
        x = x.astype(np.int32)
    return x

def isscalar(x):
    return isinstance(x, (int, float))
    
def step_str(step):
    if step < 1000:
        return f'{step}'
    elif step < 1000000:
        return f'{step/1000:.3g}k'
    else:
        return f'{step/1000000:.3g}m'

def expand_dims_match(x, target):
    """ Expands dimensions of x to match target,
    an efficient implementation of the following process 
        while len(x.shape) < len(target.shape):
            x = np.expand_dims(x, -1)
    """
    assert x.shape == target.shape[:x.ndim], (x.shape, target.shape)
    return x[(*[slice(None) for _ in x.shape], *(None,)*(target.ndim - x.ndim))]

def moments(x, axis=None, mask=None):
    if x.dtype == np.uint8:
        x = x.astype(np.int32)
    if mask is None:
        x_mean = np.mean(x, axis=axis)
        x2_mean = np.mean(x**2, axis=axis)
    else:
        if axis is None:
            axis = tuple(range(x.ndim))
        else:
            axis = (axis,) if isinstance(axis, int) else tuple(axis)
        assert mask.ndim == len(axis), (mask.shape, axis)
        # compute valid entries in x corresponding to True in mask
        n = np.sum(mask)
        if n == 0:
            return 0, 0
        # the following process is about 5x faster than np.nan*
        # expand mask to match the dimensionality of x
        mask = expand_dims_match(mask, x)
        for i in axis:
            if mask.shape[i] != 1:
                assert mask.shape[i] == x.shape[i], (
                    f'{i}th dimension of mask({mask.shape[i]}) does not match'
                    f'that of x({x.shape[i]})')
            else:
                n *= x.shape[i]
        # compute x_mean and x_std from entries in x corresponding to True in mask
        x_mask = x * mask
        x_mean = np.sum(x_mask, axis=axis) / n
        x2_mean = np.sum(x_mask**2, axis=axis) / n
    x_var = x2_mean - x_mean**2

    return x_mean, x_var
    
def standardize(x, mask=None, axis=None, epsilon=1e-8):
    if mask is not None:
        mask = expand_dims_match(mask, x)
    x_mean, x_var = moments(x, axis=axis, mask=mask)
    x_std = np.sqrt(x_var + epsilon)
    y = (x - x_mean) / x_std
    if mask is not None:
        y = np.where(mask == 1, y, x)
    return y

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def eval_str(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val

def is_main_process():
    return multiprocessing.current_process().name == 'MainProcess'

def set_global_seed(seed=42, tf=None):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf:
        tf.random.set_seed(seed)

def timeformat(t):
    return f'{t:.2e}'

def get_and_unpack(x):
    """
    This function is used to decompose a list of remote objects 
    that corresponds to a tuple of lists.

    For example:
    @ray.remote
    def f():
        return ['a', 'a'], ['b', 'b']

    get_and_unpack(ray.get([f.remote() for _ in range(2)]))
    >>> [['a', 'a', 'a', 'a'], ['b', 'b', 'b', 'b']]
    """
    list_of_lists = list(zip(*x))
    results = []
    for item_list in list_of_lists:
        tmp = []
        for item in item_list:
            tmp += item
        results.append(tmp)

    return results

def squarest_grid_size(n, more_on_width=True):
    """Calculates the size of the most squared grid for n.

    Calculates the largest integer divisor of n less than or equal to
    sqrt(n) and returns that as the width. The height is
    n / width.

    Args:
        n: The total number of images.
        more_on_width: If cannot fit in a square, put more cells on width
    Returns:
        A tuple of (height, width) for the image grid.
    """
    # the following code is useful for large n, but it is not compatible with tf.numpy_function
    # import sympy
    # divisors = sympy.divisors(n)
    # square_root = math.sqrt(n)
    # for d in divisors:
    #     if d > square_root:
    #         break

    square_root = math.ceil(np.sqrt(n))
    for d in range(square_root, n+1):
        if n // d * d == n:
            break
    h, w = int(n // d), d
    if not more_on_width:
        h, w = w, h

    return h, w

def zip_pad(*args):
    list_len = None
    for x in args:
        if isinstance(x, list) or isinstance(x, tuple):
            list_len = len(x)
            break
    assert list_len is not None
    new_args = []
    for i, x in enumerate(args):
        if not isinstance(x, list) and not isinstance(x, tuple):
            new_args.append([x] * list_len)
        else:
            new_args.append(x)

    return list(zip(*new_args))
    
def convert_indices(indices, *args):
    """ 
    convert 1d indices to a tuple of for ndarray index
    args specify the size of the first len(args) dimensions
    e.g.
    x = np.array([[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]])
    print(x.shape)
    >>> (2, 2, 2)
    indices = np.random.randint(7, size=5)
    print(indices)
    >>> [6 6 0 3 1]
    indices = convert_shape(indices, *x.shape)
    print(indices)
    >>> (array([1, 1, 0, 0, 0]), array([1, 1, 0, 1, 0]), array([0, 0, 0, 1, 1]))
    print(x[indices])
    >>> array(['b0', 'c1', 'b1', 'a1', 'c0'])
    """
    res = []
    v = indices
    for i in range(1, len(args)):
        prod = np.prod(args[i:])
        res.append(v // prod)
        v = v % prod
    res.append(v)

    return tuple(res)

def infer_dtype(dtype, precision=None):
    if precision is None:
        return dtype
    elif np.issubdtype(dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(dtype, np.uint8):
        dtype = np.uint8
    elif dtype == np.bool:
        dtype = np.bool
    else:
        dtype = None
    return dtype

def convert_dtype(value, precision=32, dtype=None, **kwargs):
    value = np.array(value, copy=False, **kwargs)
    if dtype is None:
        dtype = infer_dtype(value.dtype, precision)
    return value.astype(dtype)

def flatten_dict(**kwargs):
    """ Flatten a dict of lists into a list of dicts
    For example
    flatten_dict(lr=[1, 2], a=[10,3], b=dict(c=[2, 4], d=np.arange(1, 3)))
    >>>
    [{'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 1}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 2}}]
    """
    ks, vs = [], []
    for k, v in kwargs.items():
        ks.append(k)
        if isinstance(v, dict):
            vs.append(flatten_dict(**v))
        elif isinstance(v, (int, float)):
            vs.append([v])
        else:
            vs.append(v)

    result = []

    for k, v in itertools.product([ks], zip(*vs)):
        result.append(dict(zip(k, v)))

    return result

def product_flatten_dict(**kwargs):
    """ Flatten a dict of lists into a list of dicts
    using the Cartesian product
    For example
    product_flatten_dict(lr=[1, 2], a=[10,3], b=dict(c=[2, 4], d=np.arange(3)))
    >>>
    [{'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 0}},
    {'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 1}},
    {'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 2}},
    {'lr': 1, 'a': 10, 'b': {'c': 4, 'd': 0}},
    {'lr': 1, 'a': 10, 'b': {'c': 4, 'd': 1}},
    {'lr': 1, 'a': 10, 'b': {'c': 4, 'd': 2}},
    {'lr': 1, 'a': 3, 'b': {'c': 2, 'd': 0}},
    {'lr': 1, 'a': 3, 'b': {'c': 2, 'd': 1}},
    {'lr': 1, 'a': 3, 'b': {'c': 2, 'd': 2}},
    {'lr': 1, 'a': 3, 'b': {'c': 4, 'd': 0}},
    {'lr': 1, 'a': 3, 'b': {'c': 4, 'd': 1}},
    {'lr': 1, 'a': 3, 'b': {'c': 4, 'd': 2}},
    {'lr': 2, 'a': 10, 'b': {'c': 2, 'd': 0}},
    {'lr': 2, 'a': 10, 'b': {'c': 2, 'd': 1}},
    {'lr': 2, 'a': 10, 'b': {'c': 2, 'd': 2}},
    {'lr': 2, 'a': 10, 'b': {'c': 4, 'd': 0}},
    {'lr': 2, 'a': 10, 'b': {'c': 4, 'd': 1}},
    {'lr': 2, 'a': 10, 'b': {'c': 4, 'd': 2}},
    {'lr': 2, 'a': 3, 'b': {'c': 2, 'd': 0}},
    {'lr': 2, 'a': 3, 'b': {'c': 2, 'd': 1}},
    {'lr': 2, 'a': 3, 'b': {'c': 2, 'd': 2}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 0}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 1}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 2}}]
    """
    ks, vs = [], []
    for k, v in kwargs.items():
        ks.append(k)
        if isinstance(v, dict):
            vs.append(product_flatten_dict(**v))
        elif isinstance(v, (int, float)):
            vs.append([v])
        else:
            vs.append(v)

    result = []

    for k, v in itertools.product([ks], itertools.product(*vs)):
        result.append(dict(zip(k, v)))

    return result

def batch_dicts(x, func=np.stack):
    keys = x[0].keys()
    vals = [o.values() for o in x]
    vals = [func(v) for v in zip(*vals)]
    x = {k: v for k, v in zip(keys, vals)}
    return x

def concat_map(x):
    return tf.nest.map_structure(lambda x: np.concatenate(x), x)


def get_frame(backtrack):
    frame = inspect.currentframe()
    for _ in range(backtrack):
        frame = frame.f_back
    return frame

class TempStore:
    def __init__(self, get_fn, set_fn):
        self._get_fn = get_fn
        self._set_fn = set_fn

    def __enter__(self):
        self.state = self._get_fn()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._set_fn(self.state)
