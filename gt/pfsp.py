import random
import numpy as np

from core.typing import ModelPath
from gt.payoff import PayoffTableWithModel


class PFSP:
    def __init__(
        self, 
        p, 
        **kwargs
    ):
        self._p = p

    def __call__(
        self, 
        aid: int, 
        model: ModelPath, 
        payoff_table: PayoffTableWithModel
    ):
        """ Prioritized Fictitous Self-Play """
        model2sid = payoff_table.get_model2sid()
        sid2model = payoff_table.get_sid2model()
        sid = model2sid[aid]
        payoff = payoff_table.get_payoffs_for_agent(aid, sid=sid)
        models = []
        for i in range(len(model2sid)):
            if i == aid:
                assert model in model2sid[i], (i, model, list(model2sid[aid]))
                models.append(model)
            else:
                payoff_i = payoff.mean(axis=tuple(
                    [j for j in range(len(model2sid)) if i != j]))
                assert payoff_i.size == payoff.shape[i], (payoff_i.size, payoff.shape)
                weights_i = (1 - payoff_i[:-1]) ** self._p
                model = random.choices(sid2model[i][:-1], weights=weights_i)
                models.append(model)

        return models
