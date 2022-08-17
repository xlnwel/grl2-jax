import numpy as np


def xy2tri(y, x, return_length=False):
    """ convert relative (x, y) distance to a trignometric representation """
    x = np.where(x == 0, 1e-17, x)
    rad = np.arctan(y / x)

    if return_length:
        return np.cos(rad), np.sin(rad), np.linalg.norm([x, y], axis=0)
    else:
        return np.cos(rad), np.sin(rad)
