import numpy as np


def kl(pi1, pi2, is_action_discrete):
    if is_action_discrete:
        kl = kl_categorical(pi1, pi2)
    else:
        kl = kl_gaussian(*pi1, *pi2)
    return kl

def kl_gaussian(mu1, std1, mu2, std2, eps=1e-8):
    logstd1 = np.log(std1)
    logstd2 = np.log(std2)
    kl = np.sum(logstd2 - logstd1 - .5 
            + .5 * (std1**2 + (mu1 - mu2)**2) / (std2 + eps)**2, 
        axis=-1)
    return kl

def kl_categorical(pi1, pi2):
    logpi1 = np.log(pi1)
    logpi2 = np.log(pi2)
    kl = np.sum(pi1 * (logpi1 - logpi2), axis=-1)
    return kl

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
