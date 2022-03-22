from typing import List

from core.typing import ModelPath
from gt.func import select_sampling_strategy
from gt.payoff import PayoffWithModel


class PayoffManager:
    def __init__(
        self,
        config,
        n_agents,
        model_dir,
        name='payoff',
    ):
        self.config = config
        self.n_agents = n_agents
        self.name = name

        self._dir = model_dir
        self._path = f'{self._dir}/{self.name}.pkl'

        self.payoff = PayoffWithModel(
            n_agents=n_agents, 
            step_size=self.config.step_size, 
            payoff_dir=model_dir, 
            name=name
        )

        self.sampling_strategy = select_sampling_strategy(
            self.config.sampling_strategy
        )

        self.restore()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        elif hasattr(self.payoff, name):
            return getattr(self.payoff, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    """ Strategy Management """
    def get_all_strategies(self):
        return self.payoff.get_all_models()

    def add_strategy(self, model: ModelPath):
        """ Add a strategy for a single agent """
        self.payoff.expand(model)

    def add_strategies(self, models: List[ModelPath]):
        """ Add strategies for all agents at once """
        self.payoff.expand_all(models)

    def sample_strategies(self, aid, model: ModelPath):
        return self.sampling_strategy(
            aid, 
            model, 
            payoff=self.payoff, 
        )

    """ Payoff Management """
    def update_payoffs(self, models: List[ModelPath], scores: List[List[float]]):
        self.payoff.update(models, scores)
