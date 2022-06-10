import random
import numpy as np

from core.typing import ModelPath
from gt.payoff import PayoffTableWithModel


class PFSP:
    def __init__(
        self, 
        p, 
        threshold=.1, 
        **kwargs
    ):
        self._p = p
        self._threshold = threshold

    def __call__(
        self, 
        aid: int, 
        model: ModelPath, 
        payoff_table: PayoffTableWithModel
    ):
        """ Prioritized Fictitous Self-Play """
        model2sid = payoff_table.get_model2sid()
        sid2model = payoff_table.get_sid2model()
        sid = model2sid[aid][model]
        payoff = payoff_table.get_payoffs_for_agent(aid, sid=sid)
        models = []
        for i in range(len(model2sid)):
            if i == aid:
                assert model in model2sid[i], (i, aid, model, model2sid)
                models.append(model)
            else:
                payoff_i = payoff.mean(axis=tuple([
                    j if j < aid else j-1 
                    for j in range(len(model2sid)) 
                    if j != i and j != aid
                ]))
                assert len(payoff_i.shape) == 1, (payoff_i.size, payoff.shape)
                assert payoff_i.size == len(model2sid[i]), (payoff_i.size, payoff.shape)
                # we do not consider the most recent model here
                if np.any(payoff_i > 1) or np.any(payoff_i < 0):
                    pmax = payoff_i.max()
                    pmin = payoff_i.min()
                    pmax += (pmax - pmin) * self._threshold  # to avoid zero probability
                    payoff_i = (payoff_i - pmin) / (pmax - pmin)
                weights_i = self._compute_rank_weights(payoff_i[:-1])
                sampled_model = random.choices(sid2model[i][:-1], weights=weights_i)[0]
                models.append(sampled_model)

        return models

    def compute_opponent_distribution(
        self, 
        aid: int, 
        model: ModelPath, 
        payoff_table: PayoffTableWithModel
    ):
        payoffs, weights = self.compute_opponent_weights(
            aid, model, payoff_table)
        dists = weights / np.sum(weights, -1, keepdims=True)
        assert np.all(dists > 0), dists
        return payoffs, dists

    def compute_opponent_weights(
        self, 
        aid: int, 
        model: ModelPath, 
        payoff_table: PayoffTableWithModel
    ):
        """ Prioritized Fictitous Self-Play """
        model2sid = payoff_table.get_model2sid()
        sid = model2sid[aid][model]
        payoff = payoff_table.get_payoffs_for_agent(aid, sid=sid)
        payoffs = []
        weights = []
        for i in range(len(model2sid)):
            if i == aid:
                continue
            else:
                payoff_i = payoff.mean(axis=tuple([
                    j if j < aid else j-1 
                    for j in range(len(model2sid)) 
                    if j != i and j != aid
                ]))
                payoffs.append(payoff_i)
                assert len(payoff_i.shape) == 1, (payoff_i.size, payoff.shape)
                assert payoff_i.size == len(model2sid[i]), (payoff_i.size, payoff.shape)
                # we do not consider the most recent model here
                if np.any(payoff_i > 1) or np.any(payoff_i < 0):
                    pmax = payoff_i.max()
                    pmin = payoff_i.min()
                    if pmax == pmin:
                        weights_i = 0
                    else:
                        pmax += (pmax - pmin) * self._threshold  # to avoid zero weights
                        weights_i = (payoff_i - pmin) / (pmax - pmin)
                else:
                    weights_i = payoff_i
                weights_i = self._compute_rank_weights(weights_i)
                weights.append(weights_i)
        payoffs = np.stack(payoffs)
        weights = np.stack(weights)
        assert np.all(weights > 0), weights
        return payoffs, weights
        
    def _compute_rank_weights(self, w):
        return (1 - w) ** self._p
