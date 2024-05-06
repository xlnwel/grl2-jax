import os
import jax
from jax import random
import jax.numpy as jnp
import chex

from core.names import DEFAULT_ACTION
from core.mixin.model import update_params
from core.typing import AttrDict
from tools.file import source_file
from tools.utils import batch_dicts
from algo.ma_common.elements.model import *
from algo.masac.elements.utils import concat_along_unit_dim


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class Model(MAModelBase):
  def add_attributes(self):
    super().add_attributes()
    self.target_params = AttrDict()

  def build_nets(self):
    aid = self.config.get('aid', 0)
    data = construct_fake_data(self.env_stats, aid=aid)

    # policies for each agent
    self.params.policies = []
    policy_init, self.modules.policy = self.build_net(
      name='policy', return_init=True)
    self.rng, policy_rng, q_rng = random.split(self.rng, 3)
    self.act_rng = self.rng
    for rng in random.split(policy_rng, self.n_groups):
      self.params.policies.append(policy_init(
        rng, data.obs, data.state_reset, data.state, data.action_mask
      ))
    
    self.params.Qs = []
    q_init, self.modules.Q = self.build_net(name='Q', return_init=True)
    global_state = data.global_state[:, :, :1]
    for rng in random.split(q_rng, self.config.n_Qs):
      self.params.Qs.append(q_init(
        rng, global_state, data.action, data.state_reset, data.state
      ))
    self.params.temp, self.modules.temp = self.build_net(name='temp')

    self.sync_target_params()

  def compile_model(self):
    self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))

  @property
  def target_theta(self):
    return self.target_params

  def sync_target_params(self):
    self.target_params = self.params.copy()
    chex.assert_trees_all_close(self.params, self.target_params)

  def update_target_params(self):
    self.target_params = update_params(
      self.params, self.target_params, self.config.polyak)

  def raw_action(self, params, rng, data, evaluation=False):
    rngs = random.split(rng, self.n_groups)
    all_actions = []
    all_stats = []
    all_states = []
    for gid, (p, rng) in enumerate(zip(params.policies, rngs)):
      agent_rngs = random.split(rng, 2)
      d = data[gid]
      if self.has_rnn:
        state = d.pop('state', AttrDict())
        d = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1) , d)
        d.state = state
      else:
        state = AttrDict()
      act_outs, state.policy = self.forward_policy(p, agent_rngs[0], d, state=state.policy)
      act_dists = self.policy_dist(act_outs, evaluation)

      if evaluation:
        action = dict2AttrDict({k: ad.sample(seed=rngs[1]) for k, ad in act_dists.items()})
        stats = AttrDict()
      else:
        if len(act_dists) == 1:
          action = act_dists[DEFAULT_ACTION].sample(seed=agent_rngs[1])
          action = dict2AttrDict({DEFAULT_ACTION: action})
          stats = act_dists[DEFAULT_ACTION].get_stats('mu')
          if not self.is_action_discrete[DEFAULT_ACTION]:
            action = jnp.tanh(action)
        else:
          action = AttrDict()
          stats = AttrDict()
          rngs = random.split(rngs[1])
          for i, (k, ad) in enumerate(act_dists.items()):
            a = ad.sample(seed=rngs[i])
            action[k] = a
            k = k.replace('action_', '')
            stats.update(ad.get_stats(f'{k}_mu'))

      all_actions.append(action)
      all_stats.append(stats)

    action = batch_dicts(all_actions, func=concat_along_unit_dim)
    stats = batch_dicts(all_stats, func=concat_along_unit_dim)

    return action, stats, all_states


  """ RNN Operators """
  def get_initial_state(self, batch_size, name='default'):
    name = f'{name}_{batch_size}'
    if name in self._initial_states:
      return self._initial_states[name]
    if not self.has_rnn:
      return None
    data = construct_fake_data(self.env_stats, batch_size)
    raise NotImplementedError
    self._initial_states[name] = jax.tree_util.tree_map(jnp.zeros_like, states)

    return self._initial_states[name]


def setup_config_from_envstats(config, env_stats):
  idx = config.gid or config.aid
  config.policy.action_dim = env_stats.action_dim[idx]
  config.policy.is_action_discrete = env_stats.is_action_discrete[idx]
  config.policy.use_action_mask = env_stats.use_action_mask[idx]
  config.Q.action_dim = env_stats.action_dim[idx]
  config.Q.is_action_discrete = env_stats.is_action_discrete[idx]

  return config


def create_model(
  config, 
  env_stats, 
  name='masac', 
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
