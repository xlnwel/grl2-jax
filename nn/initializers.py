from tensorflow.keras import initializers


def constant_initializer(val):
    return initializers.Constant(val)

def get_initializer(name):
    """ 
    Return a kernel initializer by name, 
    keras.initializers.Constant(val) should not be considered here
    """
    if name == 'he_normal':
        return initializers.he_normal
    elif name == 'he_uniform':
        return initializers.he_uniform
    elif name == 'glorot_normal':
        return initializers.GlorotNormal()
    elif name == 'glorot_uniform':
        return initializers.GlorotUniform()
    else:
        raise NotImplementedError(f'Unknown initializer name {name}')
