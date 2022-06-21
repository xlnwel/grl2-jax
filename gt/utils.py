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

def compute_opponent_weights(
    aid: int, 
    model_payoff: np.ndarray, 
    n_agents: int=None, 
    prioritize_unmet=True, 
    reweight_kwargs: dict={},
):
    """ Weights for Prioritized Fictitous Self-Play """
    payoffs = []
    weights = []
    if n_agents is None:
        n_agents = len(model_payoff.shape) + 1
    assert len(model_payoff.shape) == n_agents - 1, (model_payoff.shape, n_agents)
    for i in range(n_agents):
        if i == aid:
            continue
        else:
            payoff_i = model_payoff.mean(axis=tuple([
                j if j < aid else j-1 
                for j in range(n_agents) 
                if j != i and j != aid
            ]))
            payoffs.append(payoff_i)
            assert len(payoff_i.shape) == 1, (payoff_i.size, model_payoff.shape)
            assert payoff_i.size == model_payoff.shape[0], (payoff_i.size, model_payoff.shape)
            if prioritize_unmet and np.any(np.isnan(payoff_i)):
                # prioritize unmet opponents
                weights_i = np.isnan(payoff_i)
            else:
                if np.any(payoff_i > 1) or np.any(payoff_i < 0):
                    pmax = np.nanmax(payoff_i)
                    pmin = np.nanmin(payoff_i)
                    if pmax == pmin:
                        payoff_i = np.ones_like(payoff_i, dtype=np.float32)
                    else:
                        payoff_i = (payoff_i - pmin) / (pmax - pmin)
                weights_i = 1 - payoff_i
                weights_i[np.isnan(weights_i)] = np.nanmax(weights_i)
                weights_i = compute_prioritized_weights(weights_i, **reweight_kwargs)
            weights.append(weights_i)
            # if not np.any(np.isnan(payoff_i)):
            #     assert np.all(weights_i > 0), (payoff_i, weights_i)
    payoffs = np.stack(payoffs)
    weights = np.stack(weights)

    return payoffs, weights

def compute_opponent_distribution(
    aid: int, 
    model_payoff: np.ndarray, 
    n_agents: int, 
    prioritize_unmet: bool=True, 
    reweight_kwargs: dict={},
):
    payoffs, weights = compute_opponent_weights(
        aid=aid, 
        model_payoff=model_payoff, 
        n_agents=n_agents, 
        prioritize_unmet=prioritize_unmet, 
        reweight_kwargs=reweight_kwargs
    )

    dists = weights / np.sum(weights, -1, keepdims=True)
    return payoffs, dists
