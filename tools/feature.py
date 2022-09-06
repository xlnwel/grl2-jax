import numpy as np


def one_hot(i, n):
    x = [0] * n
    x[i] = 1
    return x

def rad2tri(x):
    return np.cos(x), np.sin(x)

def xy2tri(y, x, return_length=False):
    """ convert relative (x, y) distance to a trignometric representation """
    x = np.where(x == 0, 1e-17, x)
    rad = np.arctan(y / x)

    if return_length:
        return (*rad2tri(rad), np.linalg.norm([x, y], axis=0))
    else:
        return rad2tri(rad)

def xyz2tri(x, y, z, return_length=False):
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    if return_length:
        return rad2tri(theta), rad2tri(phi), r
    else:
        return rad2tri(theta), rad2tri(phi)
