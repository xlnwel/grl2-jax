import os
import jax
from jax import random
import jax.numpy as jnp

from env.utils import get_action_mask
from core.names import DEFAULT_ACTION
from core.typing import AttrDict, dict2AttrDict
from tools.file import source_file
from algo.ma_common.elements.model import *


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class Model(MAModelBase):
  def build_nets(self):
    aid = self.config.get('aid', 0)
    data = construct_fake_data(self.env_stats, aid=aid)

    self.params.policy, self.modules.policy = self.build_net(
      data.obs, data.state_reset, data.state, data.action_mask, name='policy')
    # self.params.value, self.modules.value = self.build_net(
    #   data.global_state, data.state_reset, data.state, name='value')

  def compile_model(self):
    self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))

  def action(self, data, evaluation):
    if 'global_state' not in data:
      data.global_state = data.obs
    return super().action(data, evaluation)

  def raw_action(
    self, 
    params, 
    rng, 
    data, 
    evaluation=False, 
  ):
    rngs = random.split(rng, 3)
    state = data.pop('state', AttrDict())
    # add the sequential dimension
    if self.has_rnn:
      data = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), data)
    act_outs, state.policy = self.forward_policy(params.policy, rngs[0], data, state.policy)
    act_dists = self.policy_dist(act_outs, evaluation)

    if evaluation:
      action = {k: ad.sample(seed=rngs[1]) for k, ad in act_dists.items()}
      stats = AttrDict()
    else:
      if len(act_dists) == 1:
        action, logprob = act_dists[DEFAULT_ACTION].sample_and_log_prob(seed=rngs[1])
        action = {DEFAULT_ACTION: action}
        stats = act_dists[DEFAULT_ACTION].get_stats('mu')
        stats = dict2AttrDict(stats)
        stats.mu_logprob = logprob
      else:
        action = AttrDict()
        logprob = AttrDict()
        stats = AttrDict(mu_logprob=0)
        for k, ad in act_dists.items():
          a, lp = ad.sample_and_log_prob(seed=rngs[1])
          action[k] = a
          logprob[k] = lp
          stats.update(ad.get_stats(f'{k}_mu'))
          stats.mu_logprob = stats.mu_logprob + lp
  
    if self.has_rnn:
      # squeeze the sequential dimension
      action, stats = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, 1), (action, stats))
    if state.policy is None:
      state = None
    
    return action, stats, state

  """ RNN Operators """
  def get_initial_state(self, batch_size, name='default'):
    name = f'{name}_{batch_size}'
    if name in self._initial_states:
      return self._initial_states[name]
    if not self.has_rnn:
      return None
    data = construct_fake_data(self.env_stats, self.aid, batch_size=batch_size)
    action_mask = get_action_mask(data.action)
    state = AttrDict()
    _, state.policy = self.modules.policy(
      self.params.policy, 
      self.act_rng, 
      data.obs, 
      reset=data.state_reset, 
      action_mask=action_mask
    )
    self._initial_states[name] = jax.tree_util.tree_map(jnp.zeros_like, state)

    return self._initial_states[name]


def create_model(
  config, 
  env_stats, 
  name='ppo', 
  **kwargs
): 
  config = setup_config_from_envstats(config, env_stats)

  return Model(
    config=config, 
    env_stats=env_stats, 
    name=name, 
    **kwargs
  )


# if __name__ == '__main__':
#   from tools.yaml_op import load_config
#   from env.func import create_env
#   from tools.display import pwc
#   config = load_config('algo/zero_mr/configs/magw_a2c')
  
#   env = create_env(config.env)
#   model = create_model(config.model, env.stats())
#   data = construct_fake_data(env.stats(), 0)
#   print(model.action(model.params, data))
#   pwc(hk.experimental.tabulate(model.raw_action)(model.params, data), color='yellow')
