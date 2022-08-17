import numpy as np


def xy2tri(y, x):
    """ convert relative (x, y) distance to a trignometric representation """
    x = np.where(x == 0, 1e-17, x)
    rad = np.arctan(y / x)

    return np.cos(rad), np.sin(rad)
