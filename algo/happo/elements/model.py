import os
import jax
from jax import random
import jax.numpy as jnp

from core.typing import AttrDict, tree_slice
from jax_tools import jax_utils
from tools.file import source_file
from tools.utils import batch_dicts
from algo.ma_common.elements.model import *


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


def concat_along_unit_dim(x):
  x = jnp.concatenate(x, axis=1)
  return x


class Model(MAModelBase):
  def build_nets(self):
    aid = self.config.get('aid', 0)
    data = construct_fake_data(self.env_stats, aid=aid)

    # policies for each agent
    self.params.policies = []
    policy_init, self.modules.policy = self.build_net(
      name='policy', return_init=True)
    self.rng, policy_rng, value_rng = random.split(self.rng, 3)
    self.act_rng = self.rng
    for rng in random.split(policy_rng, self.n_groups):
      self.params.policies.append(policy_init(
        rng, data.obs, data.state_reset, data.state, data.action_mask
      ))
    
    self.params.vs = []
    value_init, self.modules.value = self.build_net(
      name='value', return_init=True)
    for rng in random.split(value_rng, self.n_groups):
      self.params.vs.append(value_init(
        rng, data.global_state, data.state_reset, data.state
      ))

  def compile_model(self):
    self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))
    self.jit_forward_policy = jax.jit(
      self.forward_policy, static_argnames=('return_state'))
    self.jit_action_logprob = jax.jit(self.action_logprob)

  def action(self, data, evaluation):
    for d in data:
      if 'global_state' not in d:
        d.global_state = d.obs
    return super().action(data, evaluation)

  def raw_action(
    self, 
    params, 
    rng, 
    data, 
    evaluation=False, 
  ):
    rngs = random.split(rng, self.n_groups)
    all_actions = []
    all_stats = []
    all_states = []
    for gid, (p, v, rng) in enumerate(zip(params.policies, params.vs, rngs)):
      agent_rngs = random.split(rng, 3)
      d = data[gid]
      state = d.pop('state', AttrDict())
      if self.has_rnn:
        d = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1) , d)
      act_out, state = self.forward_policy(p, agent_rngs[0], d, state)
      act_dist = self.policy_dist(act_out, evaluation)

      if evaluation:
        action = act_dist.sample(seed=agent_rngs[1])
        stats = AttrDict()
      else:
        action, logprob = act_dist.sample_and_log_prob(seed=agent_rngs[1])
        stats = act_dist.get_stats('mu')
        value, state.value = self.modules.value(
          v, 
          agent_rngs[2], 
          d.global_state, 
          d.state_reset, 
          state.value
        )
        stats.update({'mu_logprob': logprob, 'value': value})
      if self.has_rnn:
        action, stats = jax.tree_util.tree_map(
          lambda x: jnp.squeeze(x, 1), (action, stats))
        all_states.append(state)
      else:
        all_states = None

      all_actions.append(action)
      all_stats.append(stats)

    action = concat_along_unit_dim(all_actions)
    stats = batch_dicts(all_stats, func=concat_along_unit_dim)

    return action, stats, all_states

  def action_logprob(
    self,
    params,
    rng,
    data,
  ):
    data.state_reset, _ = jax_utils.split_data(
      data.state_reset, axis=1)
    if 'state' in data:
      data.state = tree_slice(data.state, indices=0, axis=1)
    state = data.pop('state', AttrDict())
    act_out, _ = self.forward_policy(params, rng, data, state=state)
    act_dist = self.policy_dist(act_out)
    logprob = act_dist.log_prob(data.action)

    return logprob

  def compute_value(self, data):
    @jax.jit
    def comp_value(params, rng, data):
      vs = []
      for p, d in zip(params, data):
        state = d.pop('state', AttrDict())
        if self.has_rnn:
          d = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1) , d)
        v, _ = self.modules.value(
          p, rng, 
          d.global_state, 
          d.state_reset, 
          state.value
        )
        vs.append(v)
      vs = jnp.concatenate(vs, -1)
      if self.has_rnn:
        vs = jnp.squeeze(vs, 1)
      return vs
    self.act_rng, rng = random.split(self.act_rng)
    value = comp_value(self.params.vs, rng, data)
    return value

  """ RNN Operators """
  def get_initial_state(self, batch_size, name='default'):
    name = f'{name}_{batch_size}'
    if name in self._initial_states:
      return self._initial_states[name]
    if not self.has_rnn:
      return None
    data = construct_fake_data(self.env_stats, batch_size)
    states = []
    for p, v, d in zip(self.params.policies, self.params.vs, data):
      state = AttrDict()
      _, state.policy = self.modules.policy(
        p, self.act_rng, d.obs, d.state_reset)
      _, state.value = self.modules.value(
        v, self.act_rng, d.global_state, d.state_reset)
      states.append(state)
    states = tuple(states)
    self._initial_states[name] = jax.tree_util.tree_map(jnp.zeros_like, states)

    return self._initial_states[name]


def create_model(
  config, 
  env_stats, 
  name='happo', 
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
#   data = construct_fake_data(env.stats())
#   print(model.action(model.params, data))
#   pwc(hk.experimental.tabulate(model.raw_action)(model.params, data), color='yellow')
