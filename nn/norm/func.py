from tensorflow.keras import layers


def get_norm(name):
    """ Return a normalization by name """
    if name == 'layer':
        return layers.LayerNormalization
    elif name == 'batch':
        return layers.BatchNormalization
    elif name is None or name.lower() == 'none':
        return None
    else:
        raise NotImplementedError
