import os
from typing import Dict, List
import cloudpickle
import numpy as np

from core.typing import ModelPath
from distributed.sync_sim.remote.utils import get_aid


class PayoffManager:
    def __init__(
        self,
        model_dir,
        n_agents,
        step_size,
        name='payoff'
    ):
        self._n_agents = n_agents
        self._step_size = step_size
        self._name = name

        self._dir = model_dir
        self._path = f'{self._dir}/{self._name}.pkl'

        # payoff tensors for agents
        self.payoffs = [np.zeros([0] * n_agents, dtype=np.float32) for _ in range(n_agents)]
        self.counts = [np.zeros([0] * n_agents, dtype=np.int64) for _ in range(n_agents)]
        # maps from ModelPath to strategy index
        self.model2sid: List[Dict[ModelPath, int]] = [{} for _ in range(n_agents)]
        self.restore()
    
    """ Strategy Management """
    def add_strategy(self, model: ModelPath):
        """ Add a strategy for a single agent """
        aid = get_aid(model.model_name)
        if model in self.model2sid[aid]:
            raise ValueError(f'Model({model}) is already in {list(self.model2sid[aid])}')
        self.model2sid[aid][model] = self.payoffs[aid].shape[aid]

        pad_width = [(0, 0)] * self._n_agents
        pad_width[aid] = (0, 1)
        for i in range(self._n_agents):
            self.payoffs[i] = np.pad(self.payoffs[i], pad_width)
            self.counts[i] = np.pad(self.counts[i], pad_width)

        shape = self.payoffs[0].shape
        for payoff in self.payoffs:
            assert shape == payoff.shape, (shape, payoff.shape)
    
    def add_strategies(self, models: List[ModelPath]):
        """ Add strategies for all agents at once """
        assert len(models) == self._n_agents, models
        pad_width = [(0, 1)] * self._n_agents
        for aid, model in enumerate(models):
            assert aid == get_aid(model.model_name), f'Inconsistent aids: {aid} vs {get_aid(model.model_name)}'
            if model in self.model2sid[aid]:
                raise ValueError(f'Model({model}) is already in {list(self.model2sid[aid])}')
            self.model2sid[aid][model] = self.payoffs[aid].shape[aid]

            self.payoffs[aid] = np.pad(self.payoffs[aid], pad_width)
            self.counts[aid] = np.pad(self.counts[aid], pad_width)

        shape = self.payoffs[0].shape
        for payoff in self.payoffs:
            assert shape == payoff.shape, (shape, payoff.shape)

    """ Payoff Operations """
    def update_payoffs(self, models: List[ModelPath], scores: List[List[float]]):
        assert len(models) == len(scores) == self._n_agents, (models, scores, self._n_agents)
        sids = tuple([
            m2sid[model] for m2sid, model in zip(self.model2sid, models) 
            if model in m2sid])
        assert len(sids) == self._n_agents, f'Some models are not spe'
        for payoff, count, s in zip(self.payoffs, self.counts, scores):
            if s == []:
                continue
            elif count[sids] == 0:
                payoff[sids] = sum(s) / len(s)
            elif self._step_size == 0:
                payoff[sids] = (count[sids] * payoff[sids] + sum(s)) / (count[sids] + len(s))
            else:
                payoff[sids] += self._step_size * (sum(s) - payoff[sids])
            count[sids] += len(s)

    """ Checkpoints """
    def save(self):
        with open(self._path, 'wb') as f:
            cloudpickle.dump((self.payoffs, self.counts, self.model2sid), f)

    def restore(self):
        if os.path.exists(self._path):
            with open(self._path, 'rb') as f:
                self.payoffs, self.counts, self.model2sid = cloudpickle.load(f)
