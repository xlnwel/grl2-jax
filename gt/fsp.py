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
