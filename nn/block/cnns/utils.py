import os, glob
import types
import importlib
import functools
import tensorflow as tf
from tensorflow.keras import layers

cnn_mapping = {None: tf.identity}
def register_cnn(name):
    def _thunk(func):
        cnn_mapping[name] = func
        return func
    return _thunk

def cnn(name, **kwargs):
    if name is None:
        return None
    name = name.lower()
    if name in cnn_mapping:
        return cnn_mapping[name](**kwargs)
    else:
        raise ValueError(f'Unknown CNN structure: {name}. Available cnn: {list(cnn_mapping)}')


time_dist_fn = lambda fn, *args, time_distributed=False, **kwargs: (
    layers.TimeDistributed(fn(*args, **kwargs))
    if time_distributed else
    fn(*args, **kwargs)
)

conv2d = functools.partial(time_dist_fn, layers.Conv2D)
depthwise_conv2d = functools.partial(time_dist_fn, layers.DepthwiseConv2D)
maxpooling2d = functools.partial(time_dist_fn, layers.MaxPooling2D)


def _source_file(_file_path):
    """
    Dynamically "sources" a provided file
    """
    basename = os.path.basename(_file_path)
    filename = basename.replace(".py", "")
    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, _file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)


def load_cnn(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for a `models` folder, and imports
    all the available files in there.
    """
    print('load_cnn', os.path.abspath(local_dir))
    for _file_path in glob.glob(os.path.join(
        local_dir, "cnns", "*.py")):
        """
        Sources a file expected to implement a
        custom model.

        The respective files are expected to do a
        `registry.register_env` call to ensure that
        the implemented envs are available in the
        ray registry.
        """
        _source_file(_file_path)