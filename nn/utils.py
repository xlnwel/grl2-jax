from tensorflow.keras import layers, activations, initializers


def get_activation(name):
    if name:
        return getattr(activations, name)
    else:
        return None
        
def get_norm(name):
    """ Return a normalization """
    if isinstance(name, str):
        if name == 'layer':
            return layers.LayerNormalization
        elif name == 'batch':
            return layers.BatchNormalization
        elif name is None or name.lower() == 'none':
            return None
        else:
            raise NotImplementedError
    else:
        # assume name is an normalization layer instance
        return name

def constant_initializer(val):
    return initializers.Constant(val)

def get_initializer(name):
    """ 
    Return a kernel initializer by name, 
    keras.initializers.Constant(val) should not be considered here
    """
    if name == 'he_normal':
        return initializers.he_normal()
    elif name == 'he_uniform':
        return initializers.he_uniform()
    elif name == 'glorot_normal':
        return initializers.GlorotNormal()
    elif name == 'glorot_uniform':
        return initializers.GlorotUniform()
    else:
        raise NotImplementedError(f'Unknown initializer name {name}')
