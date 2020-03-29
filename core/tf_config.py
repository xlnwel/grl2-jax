import os
import tensorflow as tf

from utility.display import pwc


def configure_gpu(idx=0):
    """Configure gpu for Tensorflow
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
            pwc(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU', 
                color='cyan')
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            pwc(e)
    else:
        pwc('No gpu is used', color='cyan')

def configure_threads(intra_num_threads, inter_num_threads):
    tf.config.threading.set_intra_op_parallelism_threads(intra_num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_num_threads)

def configure_precision(precision=16):
    if precision == 16:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

def silence_tf_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

def get_TensorSpecs(TensorSpecs, sequential=False, batch_size=None):
    """Construct a dict/list of TensorSpecs
    
    Args:
        TensorSpecs: A dict/list/tuple of arguments for tf.TensorSpec
        sequential: A boolean, if True, batch_size must be specified, and 
            the result TensorSpec will have fixed batch_size and a time dimension
        batch_size: Specifies the batch size
    Returns: 
        If TensorSpecs is a dict, return a dict of TensorSpecs with names 
        as they are in TensorSpecs. Otherwise, return a list of TensorSpecs
    """
    if sequential:
        assert batch_size, (
            f'For sequential data, please specify batch size')
        default_shape = [batch_size, None]
    else:
        default_shape = [batch_size]
    if isinstance(TensorSpecs, dict):
        name = TensorSpecs.keys()
        tensorspecs = tuple(TensorSpecs.values())
    else:
        name = None
        tensorspecs = TensorSpecs
    assert isinstance(tensorspecs, (list, tuple)), (
        'Expect tensorspecs to be a dict/list/tuple of arguments for tf.TensorSpec, '
        f'but get {TensorSpecs}\n')
    tensorspecs = [tf.TensorSpec(shape=() if s is None else default_shape+list(s), dtype=d, name=n)
         for s, d, n in tensorspecs]
    if name:
        return dict(zip(name, tensorspecs))
    else:
        return tensorspecs

def build(func, TensorSpecs, sequential=False, batch_size=None):
    """Build a concrete function of func

    Args:
        func: A function decorated by @tf.function
        TensorSpecs: A dict/list/tuple of arguments for tf.TensorSpec
        sequential: A boolean, if True, batch_size must be specified, and 
            the result TensorSpec will have fixed batch_size and a time dimension
        batch_size: Specifies the batch size
    Returns:
        A concrete function of func
    """
    ts = TensorSpecs
    while isinstance(ts, list):
        ts = ts[0]
    while isinstance(ts, dict):
        ts = tuple(ts.values())[0]
    if not isinstance(ts, tf.TensorSpec):
        TensorSpecs = get_TensorSpecs(TensorSpecs, sequential, batch_size)

    if isinstance(TensorSpecs, dict):
        return func.get_concrete_function(**TensorSpecs)
    else: 
        return func.get_concrete_function(*TensorSpecs)
    