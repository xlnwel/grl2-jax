import numpy as np

from jx.elements.actor import Actor
from jx.elements.strategy import Strategy
from tools.utils import dict2AttrDict


class RandomActor(Actor):
  def __init__(self, env, config, name='random'):
    super().__init__(config=config, model=None, name=name)
    self.env = env
    self.aid = config['aid']
  
  def __call__(self, env_out, *args, **kwargs):
    acts = self.env.random_action()
    return acts[self.aid], {}


class TBRandomActor(Actor):
  def __init__(self, env, config, name='random'):
    super().__init__(config=config, model=None, name=name)
    self.env = env
  
  def __call__(self, env_out, *args, **kwargs):
    acts = self.env.random_action()
    
    return acts[0], {}


class RandomStrategy(Strategy):
  def __init__(self, env, config, name='random'):
    config = dict2AttrDict(config)
    super().__init__(
      name=name, 
      config=config, 
      env_stats=env.stats(), 
      actor=RandomActor(env, config)
    )

  def __call__(self, env_out, *args, **kwargs):
    return self.actor(env_out)


class TBRandomStrategy(Strategy):
  def __init__(self, env, config, name='random'):
    config = dict2AttrDict(config)
    super().__init__(
      name=name, 
      config=config, 
      env_stats=env.stats(), 
      actor=TBRandomActor(env, config)
    )

  def __call__(self, env_out, *args, **kwargs):
    return self.actor(env_out)


def create_strategy(env, config, name='random'):
  if env.stats().is_simultaneous_move:
    strategy = RandomStrategy(env, config, name=name)
  else:
    strategy = TBRandomStrategy(env, config, name=name)
  return strategy
