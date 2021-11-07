import os
import logging
from typing import Union
import tensorflow as tf
from tensorflow.keras import mixed_precision

from core.log import do_logging

logger = logging.getLogger(__name__)


def configure_gpu(idx=0):
    """ Configures gpu for Tensorflow
    Args:
        idx: index(es) of PhysicalDevice objects returned by `list_physical_devices`
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # restrict TensorFlow to only use the i-th GPU
            tf.config.experimental.set_visible_devices(gpus[idx], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            do_logging(
                f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU', 
                logger=logger)
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            do_logging(e, logger=logger, level='warning')
        return True
    else:
        do_logging('No gpu is used', logger=logger, level='warning')
        return False

def configure_threads(intra_num_threads, inter_num_threads):
    tf.config.threading.set_intra_op_parallelism_threads(intra_num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_num_threads)

def configure_precision(precision=16):
    if precision == 16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

def silence_tf_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

def get_TensorSpecs(
        TensorSpecs: Union[dict, list, tuple], 
        sequential: bool=False, 
        batch_size: int=None, 
        add_batch_dim: bool=False):
    """ Construcs a dict/list of TensorSpecs
    
    Args:
        TensorSpecs: Arguments for tf.TensorSpec
        sequential: If True, batch_size must be specified, and 
            the result TensorSpec will have fixed batch_size and a time dimension
        batch_size: Specifies the batch size
        add_batch_dim: Whether implicitly add the batch dimension 
            at the leading dim
    Returns: 
        If TensorSpecs is a dict, return a dict of TensorSpecs with names 
        as they are in TensorSpecs. Otherwise, return a list of TensorSpecs
    """
    def construct(x, default_shape):
        """
        By default, default_shape is add before the first dimension of s.
        There are two ways to omit/change default_shape:
            1. to set s = None to omit default_shape, resulting in s = ()
            2. to pass an additional argument to x to override default_shape.
               Note that if s = None, this default_shape will be omitted anyway
        """
        if isinstance(x, tf.TensorSpec) \
                or isinstance(x, tf.Tensor) \
                    or isinstance(x, bool):
            return x
        elif isinstance(x, (list, tuple)):
            if hasattr(x, '_fields') or (len(x) > 1 and isinstance(x[1], tuple)):
                # x is a list/tuple of TensorSpecs, recursively construct them
                return get_TensorSpecs(x, 
                    sequential=sequential, batch_size=batch_size,
                    add_batch_dim=add_batch_dim)
            if len(x) == 1:
                s = x
                d = mixed_precision.global_policy().compute_dtype
                n = None
            elif len(x) == 2:
                s, d = x
                n = None
            elif len(x) == 3:
                s, d, n = x
            else:
                raise ValueError(f'Unknown form x: {x}')
            s = () if s is None else default_shape+list(s)
            return tf.TensorSpec(
                shape=s,
                dtype=d, 
                name=n
            )
        else:
            raise ValueError(f'Unknown form x: {x}')

    default_shape = [batch_size] if add_batch_dim else []
    if sequential:
        assert batch_size, (
            f'For sequential data, please specify batch size')
        default_shape += [None]
    if isinstance(TensorSpecs, dict):
        name = TensorSpecs.keys()
        tensorspecs = tuple(TensorSpecs.values())
    else:
        name = None
        tensorspecs = TensorSpecs
    assert isinstance(tensorspecs, (list, tuple)), (
        'Expect tensorspecs to be a dict/list/tuple of arguments for tf.TensorSpec, '
        f'but get {TensorSpecs}\n')
    tensorspecs = [construct(x, default_shape) for x in tensorspecs]
    if name:
        return dict(zip(name, tensorspecs))
    elif isinstance(TensorSpecs, tuple) and hasattr(TensorSpecs, '_fields'):
        return type(TensorSpecs)(*tensorspecs)
    else:
        return type(TensorSpecs)(tensorspecs)

def build(func, TensorSpecs, sequential=False, batch_size=None, print_terminal_info=False):
    """ Builds a concrete function of func, initializing all variables

    Args:
        func: A function decorated by @tf.function
        TensorSpecs: A dict/list/tuple of arguments for tf.TensorSpec
        sequential: A boolean, if True, batch_size must be specified, and 
            the result TensorSpec will have fixed batch_size and a time dimension
        batch_size: Specifies the batch size
    Returns:
        A concrete function of func
    """
    TensorSpecs = get_TensorSpecs(TensorSpecs, sequential, batch_size)
    level = 'print' if print_terminal_info else 'info'
    do_logging(
        f'{func.__name__} is built with TensorSpecs:',
        logger=logger,
        level=level)
    do_logging(TensorSpecs, prefix='\t', logger=logger, level=level)
    if isinstance(TensorSpecs, dict):
        return func.get_concrete_function(**TensorSpecs)
    else: 
        return func.get_concrete_function(*TensorSpecs)
    