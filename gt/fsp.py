import numpy as np

from core.typing import ModelPath
from gt.payoff import PayoffTableWithModel


class FSP:
    def __init__(self, **kwargs):
        pass

    def __call__(
        self, 
        aid: int, 
        model: ModelPath, 
        payoff_table: PayoffTableWithModel
    ):
        """ Fictitious Self-Play """
        model2sid = payoff_table.get_model2sid()
        sid2model = payoff_table.get_sid2model()
        models = []
        for i in range(len(model2sid)):
            if i == aid:
                assert model in model2sid[i], (i, model, list(model2sid[aid]))
                models.append(model)
            else:
                n_trained_strategies = len(model2sid[i]) - 1
                sid = np.random.randint(n_trained_strategies)
                models.append(sid2model[i][sid])

        return models

    def compute_opponent_distribution(
        self, 
        aid: int, 
        model: ModelPath, 
        payoff_table: PayoffTableWithModel
    ):
        model2sid = payoff_table.get_model2sid()
        sid = model2sid[aid][model]
        payoff = payoff_table.get_payoffs_for_agent(aid, sid=sid)
        payoffs = []
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
        payoffs = np.stack(payoffs)
        dist = np.ones_like(payoffs)
        dist /= np.sum(dist, -1, keepdims=True)
        return payoffs, dist