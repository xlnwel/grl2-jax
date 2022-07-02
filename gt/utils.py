from typing import List, Tuple
import numpy as np


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

def compute_prioritized_weights(w, type: str, p: float=None, threshold=0):
    if type == 'uniform':
        return np.ones_like(w)
    elif type == 'poly':            # prioritize large w
        w = np.maximum(w, threshold) if threshold else w
        return w ** p
    else:
        raise NotImplementedError(f'Unknown type {type}')

def get_opponent_payoffs(
    aid: int, 
    model_payoff: np.ndarray, 
    n_agents: int=None, 
):
    payoffs = []
    if n_agents is None:
        n_agents = len(model_payoff.shape) + 1
    assert len(model_payoff.shape) == n_agents - 1, (model_payoff.shape, n_agents)
    for i in range(n_agents):
        if i == aid:
            continue
        else:
            p = np.nanmean(model_payoff, axis=tuple([
                j if j < aid else j-1 
                for j in range(n_agents) 
                if j != i and j != aid
            ]))
            payoffs.append(p)
    payoffs = np.stack(payoffs)

    return payoffs

def compute_opponent_weights(
    opp_payoff: np.ndarray, 
    prioritize_unmet=True, 
    reweight_kwargs: dict={},
    filter_recent=True
):
    """ Weights for Prioritized Fictitous Self-Play """
    weights = []
    for p in opp_payoff:
        assert p.shape != (), p
        if prioritize_unmet and np.any(
                np.isnan(p[:-1] if filter_recent else p)):
            # prioritize unmet opponents
            weights_i = np.isnan(p)
        else:
            if np.any(p > 1) or np.any(p < 0):
                pmax = np.nanmax(p)
                pmin = np.nanmin(p)
                if pmax == pmin:
                    p = np.ones_like(p, dtype=np.float32)
                else:
                    p = (p - pmin) / (pmax - pmin)
            weights_i = 1 - p
            weights_i[np.isnan(weights_i)] = np.nanmax(weights_i)
            weights_i = compute_prioritized_weights(weights_i, **reweight_kwargs)
        weights.append(weights_i)
        # if not np.any(np.isnan(p)):
        #     assert np.all(weights_i > 0), (p, weights_i)
    weights = np.stack(weights)

    return weights

def compute_opponent_distribution(weights):
    dists = weights / np.nansum(weights, -1, keepdims=True)
    return dists
