import os
import jax
from jax import random
import jax.numpy as jnp
import chex

from core.mixin.model import update_params
from core.names import DEFAULT_ACTION
from core.typing import AttrDict
from tools.file import source_file
from algo.ma_common.elements.model import *


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class Model(MAModelBase):
  def add_attributes(self):
    super().add_attributes()
    self.target_params = AttrDict()

  def build_nets(self):
    aid = self.config.get('aid', 0)
    data = construct_fake_data(self.env_stats, aid=aid)

    self.params.policy, self.modules.policy = self.build_net(
      data.obs, data.state_reset, data.state, data.action_mask, name='policy')
    self.rng, q_rng = random.split(self.rng, 2)
    self.act_rng = self.rng
    
    self.params.Qs = []
    q_init, self.modules.Q = self.build_net(name='Q', return_init=True)
    global_state = data.global_state[:, :, :1]
    for rng in random.split(q_rng, self.config.n_Qs):
      self.params.Qs.append(q_init(
        rng, global_state, data.action, 
        data.state_reset, data.state
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
    rngs = random.split(rng, 2)
    if self.has_rnn:
      state = data.pop('state', AttrDict())
      data = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1) , data)
      data.state = state
    else:
      state = AttrDict()
    act_out, state.policy = self.forward_policy(
      params.policy, rngs[0], data, state=state.policy)
    act_dists = self.policy_dist(act_out, evaluation)

    if evaluation:
      action = {k: ad.sample(seed=rngs[1]) for k, ad in act_dists.items()}
      stats = None
    else:
      if len(act_dists) == 1:
        action = act_dists[DEFAULT_ACTION].sample(seed=rngs[1])
        action = {DEFAULT_ACTION: action}
        stats = act_dists[DEFAULT_ACTION].get_stats('mu')
      else:
        action = AttrDict()
        stats = AttrDict()
        rngs = random.split(rngs[1])
        for i, (k, ad) in enumerate(act_dists.items()):
          action[k] = ad.sample(seed=rngs[i])
          stats.update(ad.get_stats(f'{k}_mu'))
    
    for k, v in action.items():
      if not self.is_action_discrete[k]:
        action[k] = jnp.tanh(v)
    if self.has_rnn:
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
    data = construct_fake_data(self.env_stats, batch_size)
    state = AttrDict()
    _, state.policy = self.modules.policy(self.params.policy, self.act_rng, data.obs)
    state.qs = []
    for q_params in self.params.qs:
      _, q_state = self.modules.Q(q_params, self.act_rng, data.global_state)
      state.qs.append(q_state)
    self._initial_states[name] = jax.tree_util.tree_map(jnp.zeros_like, state)

    return self._initial_states[name]


def setup_config_from_envstats(config, env_stats):
  aid = config.aid
  config.policy.action_dim = env_stats.action_dim[aid]
  config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
  config.Q.is_action_discrete = env_stats.is_action_discrete[aid]
  # for k in env_stats.is_action_discrete[aid]:
  #   config.Q.out_size[k] = env_stats.action_dim[aid][k] if v else 1
  config.Q.out_size = 1

  return config


def create_model(
  config, 
  env_stats, 
  name='sac', 
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
