from typing import List, Union

from tools.log import do_logging
from core.typing import ModelPath, AttrDict
from game.func import select_sampling_strategy
from game.payoff import PayoffTableWithModel, SelfPlayPayoffTableWithModel
from tools.utils import dict2AttrDict


class PayoffManager:
  def __init__(
    self,
    config: AttrDict,
    n_agents: int, 
    model_dir: str,
    self_play: bool=False, 
    name='payoff',
  ):
    self.config = dict2AttrDict(config, to_copy=True)
    self.n_agents = n_agents
    self.self_play = self_play
    self.name = name

    if self.self_play:
      self.payoff_table = SelfPlayPayoffTableWithModel(
        step_size=self.config.step_size, 
        dir=model_dir, 
        name=name
      )
    else:
      self.payoff_table = PayoffTableWithModel(
        n_agents=self.n_agents, 
        step_size=self.config.step_size, 
        dir=model_dir, 
        name=name
      )

    self.sampling_strategy = select_sampling_strategy(
      **self.config.sampling_strategy,
    )

  def size(self, aid: int=0):
    if self.self_play:
      return self.payoff_table.size()
    else:
      return self.payoff_table.size(aid)

  def __contains__(self, model: ModelPath):
    return model in self.payoff_table

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
    if self.self_play:
      assert aid is None or aid in (0, 1), aid
      self.payoff_table.expand(model)
      do_logging(f'Adding strategy ({model}) to payoff table', color='green')
    else:
      self.payoff_table.expand(model, aid=aid)
      do_logging(f'Adding strategy ({model}) for agent ({aid}) to payoff table', color='green')

  def add_strategies(self, models: List[ModelPath]):
    """ Add strategies for all agents at once """
    if self.self_play:
      assert len(models) == 1, len(models)
      self.payoff_table.expand(models[0])
    else:
      self.payoff_table.expand_all(models)
    do_logging(f'Adding strategies {models} to payoff table', color='green')

  """ Payoff Management """
  def update_payoffs(
    self, 
    models: List[ModelPath], 
    scores: Union[List[float], List[List[float]]]
  ):
    assert len(models) == self.n_agents, (models, self.n_agents)
    self.payoff_table.update(models, scores)

  def get_payoffs_for_model(self, aid: int, model: ModelPath):
    if self.self_play:
      assert aid == 0, aid
      payoffs = self.payoff_table.get_payoffs(model=model)
    else:
      payoffs = self.payoff_table.get_payoffs_for_agent(aid, model=model)
    return payoffs

  def compute_opponent_distribution(
    self, 
    aid: int, 
    model: ModelPath, 
    prioritize_unmet: bool=True
  ):
    """ Get the distribution of payoffs for model of the agent of aid """
    if self.self_play:
      assert aid == 0, aid
      payoffs, weights, dists = self.sampling_strategy(
        aid, 
        self.payoff_table.get_payoffs(model=model), 
        2, 
        prioritize_unmet, 
        filter_recent=True
      )
    else:
      payoffs, weights, dists = self.sampling_strategy(
        aid, 
        self.payoff_table.get_payoffs_for_agent(aid, model=model), 
        self.n_agents, 
        prioritize_unmet, 
        filter_recent=True
      )
    return payoffs, weights, dists
