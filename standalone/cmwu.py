""" 
Validate the convergence of a set of strategies following the update rule defined by MWU_f
"""
import numpy as np


def payoff(*, p1=None, p2=None):
    if p1 is None:
        p = np.matmul(ps1, p2)
    else:
        p = np.matmul(p1.T, ps2)

    return p

def get_random_pi():
    p1 = np.random.uniform(0, 1, A)
    p1 = p1 / np.sum(p1)
    return p1

def get_prob(x):
    return x / np.sum(x)

def compute_obj(q, x, p):
    return np.sum(eta * x * q - x * (np.log(x) - np.log(p)))


for _ in range(10000):
    A = np.random.randint(2, 10)
    V = 10

    ps1 = np.random.uniform(0, V, (A, A))
    ps2 = np.random.uniform(0, V, (A, A))
    eta = 1 / V
    p1 = get_random_pi()
    p2 = get_random_pi()
    pi1 = get_random_pi()
    pi2 = get_random_pi()

    for i in range(100000):
        q1 = payoff(p2=pi2)
        q2 = payoff(p1=pi1)
        new_pi1 = get_prob(p1 * np.exp(eta * q1))
        new_pi2 = get_prob(p2 * np.exp(eta * q2))
        if np.all(np.abs(new_pi1 - pi1) < 1e-7) and np.all(np.abs(new_pi2 - pi2) < 1e-7):
            break
        pi1 = new_pi1
        pi2 = new_pi2
#     print(i)
    q1 = payoff(p2=pi2)
    q2 = payoff(p1=pi1)
    tp1 = get_prob(p1 * np.exp(eta * q1))
    tp2 = get_prob(p2 * np.exp(eta * q2))

    np.testing.assert_allclose(pi1, tp1, 1e-3)
    np.testing.assert_allclose(pi2, tp2, 1e-3)

    pv1 = compute_obj(q1, pi1, p1)
    pv2 = compute_obj(q2, pi2, p2)

    pi1_pt = get_prob(pi1+np.random.uniform(0, .1, pi1.shape))
    pi2_pt = get_prob(pi2+np.random.uniform(0, .1, pi2.shape))

    pv1_pt = compute_obj(q1, pi1_pt, p1)
    pv2_pt = compute_obj(q2, pi2_pt, p2)

    assert pv1_pt <= pv1, (pv1_pt, pv1)
    assert pv2_pt <= pv2, (pv2_pt, pv2)
