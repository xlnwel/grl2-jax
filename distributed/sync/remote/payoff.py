from typing import List

from core.typing import ModelPath
from gt.func import select_sampling_strategy
from gt.payoff import PayoffTableWithModel


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

        self.payoff_table = PayoffTableWithModel(
            n_agents=n_agents, 
            step_size=self.config.step_size, 
            payoff_dir=model_dir, 
            name=name
        )

        self.sampling_strategy = select_sampling_strategy(
            **self.config.sampling_strategy,
        )

        self.restore()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        elif hasattr(self.payoff_table, name):
            return getattr(self.payoff_table, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    """ Strategy Management """
    def get_all_strategies(self):
        return self.payoff_table.get_all_models()

    def add_strategy(self, model: ModelPath):
        """ Add a strategy for a single agent """
        self.payoff_table.expand(model)

    def add_strategies(self, models: List[ModelPath]):
        """ Add strategies for all agents at once """
        self.payoff_table.expand_all(models)

    def sample_strategies(self, aid, model: ModelPath):
        return self.sampling_strategy(
            aid, 
            model=model, 
            payoff_table=self.payoff_table, 
        )

    """ Payoff Management """
    def update_payoffs(
        self, 
        models: List[ModelPath], 
        scores: List[List[float]]
    ):
        self.payoff_table.update(models, scores)

    def get_opponent_distribution(
        self, 
        aid: int, 
        model: ModelPath, 
    ):
        payoffs, dist = self.sampling_strategy.compute_opponent_distribution(
            aid, model, self.payoff_table)
        return payoffs, dist
