from typing import List

from core.typing import ModelPath
from gt.func import select_sampling_strategy
from gt.payoff import PayoffTableWithModel
from utility.typing import AttrDict
from utility.utils import dict2AttrDict


class PayoffManager:
    def __init__(
        self,
        config: AttrDict,
        n_agents: int, 
        model_dir: str,
        name='payoff',
    ):
        self.config = dict2AttrDict(config, to_copy=True)
        self.n_agents = n_agents
        self.name = name

        self._dir = model_dir
        self._path = f'{self._dir}/{self.name}.pkl'

        self.payoff_table = PayoffTableWithModel(
            n_agents=self.n_agents, 
            step_size=self.config.step_size, 
            payoff_dir=model_dir, 
            name=name
        )

        self.sampling_strategy = select_sampling_strategy(
            **self.config.sampling_strategy,
        )

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        elif hasattr(self.payoff_table, name):
            return getattr(self.payoff_table, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    """ Strategy Management """
    def get_all_strategies(self):
        return self.payoff_table.get_all_models()

    def add_strategy(self, model: ModelPath, aid=None):
        """ Add a strategy for a single agent """
        self.payoff_table.expand(model, aid=aid)

    def add_strategies(self, models: List[ModelPath]):
        """ Add strategies for all agents at once """
        self.payoff_table.expand_all(models)

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
        prioritize_unmet: bool=True
    ):
        payoffs, dist = self.sampling_strategy(
            aid, 
            self.payoff_table.get_payoffs_for_agent(aid, model=model), 
            self.n_agents, 
            prioritize_unmet
        )
        return payoffs, dist
