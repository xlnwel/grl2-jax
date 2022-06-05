import numpy as np


def tsallis_log(p, tsallis_q):
    if tsallis_q == 1:
        return np.log(p)
    else:
        return (p**(1-tsallis_q) - 1) / (1 - tsallis_q)

def tsallis_exp(p, tsallis_q):
    if tsallis_q == 1:
        return np.exp(p)
    else:
        return np.maximum(
            0, 1 + (1-tsallis_q) * p)**(1 / (1-tsallis_q))
