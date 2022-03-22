from typing import List, Tuple
import numpy as np

from core.typing import ModelPath


def compute_utility(payoff, strategies, left2right=False):
#     assert payoff.ndim == len(strategies), (payoff.shape, len(strategies))
    if left2right:
        payoff = payoff.transpose(tuple(reversed(range(payoff.ndim))))
        for s in strategies:
            payoff = payoff @ s
        payoff = payoff.transpose(tuple(reversed(range(payoff.ndim))))
    else:
        for s in reversed(strategies):
            payoff = payoff @ s
    return payoff

