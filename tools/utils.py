import ast
import collections
import inspect
import itertools
import math
import multiprocessing
import numpy as np

from core.typing import AttrDict, ModelPath, dict2AttrDict


def is_empty(x):
    if isinstance(x, dict):
        if x == {}:
            return True
        for v in x.values():
            if not is_empty(v):
                return False
    elif isinstance(x, (list, tuple)):
        if x in [[], ()]:
            return True
        for v in x:
            if not is_empty(v):
                return False
    elif x is None:
        return True
    else:
        return False

def deep_update(source: dict, target: dict):
    for k, v in target.items():
        if isinstance(v, collections.abc.Mapping):
            assert k in source, f'{k} does not exist in {source}'
            assert isinstance(source[k], collections.abc.Mapping), \
                f'Inconsistent types: {type(v)} vs {type(source[k])}'
            source[k] = deep_update(source.get(k, {}), v)
        else:
            source[k] = v
    return source

def add_prefix(s, prefix):
    if prefix is not None:
        s = f'{prefix}/{s}'
    return s

def add_suffix(s, suffix):
    if suffix is not None:
        s = f'{s}/{suffix}'
    return s

def flatten_dict(d: dict, prefix=None, suffix=None):
    result = AttrDict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v, suffix=suffix)
            for kk, vv in v.items():
                result[f'{k}/{kk}'] = vv
        else:
            k = add_suffix(k, suffix)
            result[k] = v
    if prefix is not None:
        old_result = result
        result = {}
        for k, v in old_result.items():
            result[add_prefix(k, prefix)] = v

    return result

def recursively_remove(d: dict, keys: list):
    for k in keys:
        if k in d:
            del d[k]
    for k, v in d.items():
        if isinstance(v, dict):
            recursively_remove(v, keys)
    return d

def str2int(x):
    if isinstance(x, str):
        try:
            x = float(x)
        except:
            pass
    if isinstance(x, float) and x == int(x):
        x = int(x)
    return x

def eval_str_list(x):
    if isinstance(x, (list, tuple)):
        return [eval_str_list(v) for v in x]
    else:
        return str2int(x)

def eval_config(config):
    for k, v in config.items():
        if k == 'model_name':
            continue
        if isinstance(v, dict):
            config[k] = eval_config(v)
        elif isinstance(v, (list, tuple)):
            config[k] = eval_str_list(v)
        else:
            config[k] = str2int(v)
    return config

def add_attr(
    obj, 
    attrs,
):
    for k, v in attrs.items():
        setattr(obj, k, v)

def config_attr(
    obj, 
    config: dict, 
    filter_dict: bool=True, 
    config_as_attr: bool=True, 
    private_attr: bool=False
):
    """ Add values in config as attributes of obj

    Args:
        obj: the target object to which we add attributes
        config: values associated to uppercase keys
            are added as public attributes, while those
            associated to lowercase keys are added as
            private attributes
        filter_dict: whether to omit dictionaries
        config_as_attr: whether to set the config as an attribute
        config_as_attr: whether to take configurations as private attributes
    """
    config = eval_config(config)
    config = dict2AttrDict(config)
    if config_as_attr:
        setattr(obj, 'config', config)
    for k, v in config.items():
        if filter_dict and isinstance(v, dict):
            continue
        if private_attr and k.islower():
            k = f'_{k}'
        if isinstance(v, str) and k != 'root_dir' and k != 'model_name':
            try:
                v = float(v)
            except:
                pass
        if isinstance(v, float) and v == int(v):
            v = int(v)
        setattr(obj, k, v)
    return config


def set_path(config, model_path: ModelPath, max_layer=1):
    return modify_config(
        config, 
        max_layer=max_layer,
        root_dir=model_path.root_dir, 
        model_name=model_path.model_name)


def modify_config(
    config, 
    curr_layer=0, 
    max_layer=1, 
    overwrite_existed_only=False, 
    **kwargs
):
    for k, v in kwargs.items():
        if not overwrite_existed_only or k in config:
            config[k] = v
    if curr_layer < max_layer:
        for k, sub in config.items():
            if isinstance(sub, dict):
                config[k] = modify_config(sub, curr_layer+1, max_layer, overwrite_existed_only, **kwargs)
    return config


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

def expand_dims_match(x: np.ndarray, target: np.ndarray):
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
    x_var = np.maximum(x_var, 0)

    return x_mean, x_var
    
def standardize(x, zero_center=True, mask=None, axis=None, epsilon=1e-8):
    if mask is not None:
        mask = expand_dims_match(mask, x)
    x_mean, x_var = moments(x, axis=axis, mask=mask)
    x_std = np.sqrt(x_var + epsilon)
    if zero_center:
        x = (x - x_mean)
    y = x / x_std
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
        if isinstance(val, float) and val == int(val):
            val = int(val)
    except:
        pass
    return val

def is_main_process():
    return multiprocessing.current_process().name == 'MainProcess'

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

def positional_encodings(n_pos, size, base=10000):
    denominator = base ** (2 / size)
    pos_encodings = [
        [(np.sin if i % 2 == 0 else np.cos)(pos/denominator**(i//2))
            for i in range(size)]
        for pos in range(1, n_pos+1)]
    return np.array(pos_encodings)

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
    print(x[indices])key
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
    elif dtype == bool:
        dtype = bool
    else:
        dtype = None
    return dtype

def convert_dtype(value, precision=32, dtype=None, **kwargs):
    value = np.array(value, copy=False, **kwargs)
    if dtype is None:
        dtype = infer_dtype(value.dtype, precision)
    return value.astype(dtype)

def dol2lod(kwargs):
    """ Convert a dict of lists into a list of dicts
    For example
    dol2lod(lr=[1, 2], a=[10,3], b=dict(c=[2, 4], d=np.arange(1, 3)))
    >>>
    [{'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 1}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 2}}]
    """
    ks, vs = [], []
    for k, v in kwargs.items():
        ks.append(k)
        if isinstance(v, dict):
            vs.append(dol2lod(v))
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
    res = AttrDict()
    
    for k, v in x[0].items():
        if isinstance(v, dict):
            v = batch_dicts([xx[k] for xx in x], func=func)
            if v:
                res[k] = v
        elif v is None:
            continue
        elif hasattr(v, '_fields'):
            res[k] = type(v)(*[
                batch_dicts([getattr(xx[k], tk) for xx in x]) 
                if isinstance(tk, dict) else 
                func([getattr(xx[k], tk) for xx in x]) 
                for tk in v._fields
            ])
        else:
            res[k] = func([xx[k] for xx in x])

    return res

def stack_data_with_state(buffer, keys=None, seq_axis=1):
    if keys is None:
        keys = buffer.keys()

    data = AttrDict()
    for k in keys:
        if k not in buffer:
            continue
        if k == 'state':
            if isinstance(buffer[k][0], dict):
                # state is a dictionary, e.g., both actor and critic have their states
                v = AttrDict()
                for name in buffer[k][0].keys():
                    if hasattr(buffer[k][0][name], '_fields'):
                        t = type(buffer[k][0][name])
                        v[name] = [d[name] for d in buffer[k]]
                        v[name] = t(*[np.stack(x, seq_axis) for x in zip(*v[name])])
                    else:
                        v[name] = np.stack([x[name] for x in buffer[k]], seq_axis)
            else:
                # state is a single namedtuple
                if hasattr(buffer[k][0], '_fields'):
                    t = type(buffer[k][0])
                    v = t(*[np.stack(x, seq_axis) for x in zip(*buffer[k])])
                else:
                    v = np.stack([x for x in buffer[k]], seq_axis)
        else:
            v = np.stack(buffer[k], seq_axis)
        data[k] = v

    return data

def convert_batch_with_func(data, func=np.stack):
    if data != []:
        if isinstance(data[0], (np.ndarray, int, float, np.floating, np.integer)):
            data = func(data)
        elif isinstance(data[0], dict):
            data = batch_dicts(data, func)
        else:
            data = list(data)
    return data

def prefix_name(terms, name, filter=[]):
    if name is not None:
        new_terms = AttrDict()
        for k, v in terms.items():
            if k in filter:
                new_terms[k] = v
            elif '/' not in k:
                new_terms[f'{name}/{k}'] = v
            else:
                new_terms[k] = v
        return new_terms
    return terms

def get_frame(backtrack):
    frame = inspect.currentframe()
    for _ in range(backtrack):
        frame = frame.f_back
    return frame

