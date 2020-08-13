import functools
import tensorflow as tf
from tensorflow.keras import layers

cnn_mapping = {None: tf.identity}
def register_cnn(name):
    def _thunk(func):
        cnn_mapping[name] = func
        return func
    return _thunk


time_dist_fn = lambda fn, *args, time_distributed=False, **kwargs: (
    layers.TimeDistributed(fn(*args, **kwargs))
    if time_distributed else
    fn(*args, **kwargs)
)

conv2d = functools.partial(time_dist_fn, layers.Conv2D)
depthwise_conv2d = functools.partial(time_dist_fn, layers.DepthwiseConv2D)
maxpooling2d = functools.partial(time_dist_fn, layers.MaxPooling2D)
