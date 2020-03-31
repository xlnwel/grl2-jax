import numpy as np
from tensorflow.keras import layers, activations, initializers


def get_activation(name):
    if name.lower() == 'none':
        return None
    activations.get(name)
        
def get_norm(name):
    """ Return a normalization """
    if isinstance(name, str):
        if name == 'layer':
            return layers.LayerNormalization
        elif name == 'batch':
            return layers.BatchNormalization
        elif name.lower() == 'none':
            return None
        else:
            raise NotImplementedError
    else:
        # assume name is an normalization layer instance
        return name

def constant_initializer(val):
    return initializers.Constant(val)

def get_initializer(name, **kwargs):
    """ 
    Return a kernel initializer by name
    """
    if name.lower() == 'none':
        return None
    elif name.lower() == 'orthogonal':
        return initializers.orthogonal(kwargs.get('gain', 1.))
    return initializers.get(name)

def ortho_init(scale=1.0):
    """ 
    A reproduction of tf...Orthogonal, originally from openAI baselines
    """
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init