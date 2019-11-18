from tensorflow import keras


def he_initializer(scale=2., distribution='truncated_normal', seed=None):
    """ he initializer """
    return keras.initializers.VarianceScaling(scale=scale, mode='fan_in', distribution=distribution, seed=seed)

def glorot_initializer(scale=1., distribution='truncated_normal', seed=None):
    """ glorot initializer """
    return keras.initializers.VarianceScaling(scale=scale, mode='fan_avg', distribution=distribution, seed=seed)

def constant_initializer(val):
    return keras.initializers.Constant(val)

def get_initializer(name):
    """ 
    Return a kernel initializer by name, 
    keras.initializers.Constant(val) should not be considered here
    """
    if name == 'he':
        return he_initializer
    elif name == 'glorot':
        return glorot_initializer
    else:
        raise NotImplementedError(f'Unknown initializer name {name}')
