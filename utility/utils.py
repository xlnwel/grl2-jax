import os, random
import ast
import os.path as osp
import math
import multiprocessing
import numpy as np


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class Every:
    def __init__(self, period, start=0):
        self._period = period
        self._next = start
    
    def __call__(self, step):
        if step >= self._next:
            while step >= self._next:
                self._next += self._period
            return True
        return False

    def step(self):
        return self._next - self._period

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, axis=None, epsilon=1e-8, clip=10):
        """ Compute running mean and std from data
        Args:
            axis: axis along which we compute mean and std from incoming data. 
                If it's None, we only receive a sample at a time
        """
        if isinstance(axis, tuple):
            assert axis == tuple(range(np.min(axis), np.max(axis) + 1))
        self._axis = axis
        self._shape_slice = np.s_[np.min(self._axis): np.max(self._axis) + 1]
        self._mean = None
        self._var = None
        self._epsilon = epsilon
        self._count = epsilon
        self._clip = clip

    def update(self, x, mask=None):
        if self._axis is None:
            assert mask is None
            batch_mean, batch_var, batch_count = x, 0, 1
        else:
            batch_mean, batch_std = moments(x, self._axis, mask)
            batch_count = np.prod(x.shape[self._shape_slice]) if mask is None else np.sum(mask)
            batch_var = np.square(batch_std)
        if batch_count > 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if self._count == self._epsilon:
            self._mean = batch_mean
            self._var = batch_var
            self._count = batch_count
        else:
            delta = batch_mean - self._mean
            total_count = self._count + batch_count

            new_mean = self._mean + delta * batch_count / total_count
            # no minus one here to be consistent with np.std
            m_a = self._var * self._count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + np.square(delta) * self._count * batch_count / (self._count + batch_count)
            assert np.all(np.isfinite(M2)), f'M2: {M2}'
            new_var = M2 / (self._count + batch_count)
            self._mean = new_mean
            self._var = new_var
            self._count += batch_count

    def normalize(self, x, subtract_mean=True):
        assert not np.isinf(np.std(x)), f'{np.min(x)}\t{np.max(x)}'
        if subtract_mean:
            x = x - self._mean
        x = np.clip(x / (np.sqrt(self._var) + self._epsilon), -self._clip, self._clip)
        return x.astype(np.float32)

class TempStore:
    def __init__(self, get_fn, set_fn):
        self._get_fn = get_fn
        self._set_fn = set_fn

    def __enter__(self):
        self.state = self._get_fn()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._set_fn(self.state)

def to_int(s):
    return int(float(s))
    
def isscalar(x):
    return isinstance(x, (int, float))
    
def step_str(step):
    if step < 1000:
        return f'{step}'
    elif step < 1000000:
        return f'{step/1000:.3g}k'
    else:
        return f'{step/1000000:.3g}m'

def moments(x, axis=None, mask=None):
    if mask is None:
        x_mean = np.mean(x, axis=axis)
        x_std = np.std(x, axis=axis)
    else:
        if axis is None:
            axis = tuple(range(x.ndim))
        elif mask is not None:
            axis = (axis,) if isinstance(axis, int) else tuple(axis)
        # expand mask to match the dimensionality of x
        while len(mask.shape) < len(x.shape):
            mask = np.expand_dims(mask, -1)
        # compute valid entries in x corresponding to True in mask
        n = np.sum(mask)
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
        x_std = np.sqrt(np.sum(mask * (x_mask - x_mean)**2, axis=axis) / n)
    
    return x_mean, x_std
    
def standardize(x, axis=None, epsilon=1e-8, mask=None):
    if mask is not None:
        while len(mask.shape) < len(x.shape):
            mask = np.expand_dims(mask, -1)
    x_mean, x_std = moments(x, axis=axis, mask=mask)
    x = (np.array(x, copy=False) - x_mean) / (x_std + epsilon)
    if mask is not None:
        x *= mask
    return x

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

def check_make_dir(path):
    _, ext = osp.splitext(path)
    if ext: # if path is a file path, extract its directory path
        path, _ = osp.split(path)

    if not os.path.isdir(path):
        os.mkdir(path)

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
      raise NotImplementedError(dtype)
    return dtype

def convert_dtype(value, precision=32, dtype=None, **kwargs):
    value = np.array(value, **kwargs)
    if dtype is None:
        dtype = infer_dtype(value.dtype, precision)
    return value.astype(dtype)
