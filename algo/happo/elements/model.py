import os
import jax
from jax import random
import jax.numpy as jnp

from env.utils import get_action_mask
from core.names import DEFAULT_ACTION, TRAIN_AXIS
from core.typing import AttrDict, tree_slice
from nn.utils import reset_weights
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
    self.policy_rnn_init = self.config.policy.pop('rnn_init', None)
    self.value_rnn_init = self.config.value.pop('rnn_init', None)

    # policies for each agent
    prev_info = jnp.concatenate([v for v in data.prev_info.values()], -1) \
      if self.config.use_prev_info and data.prev_info else None
    self.params.policies = []
    policy_init, self.modules.policy = self.build_net(
      name='policy', return_init=True)
    self.rng, policy_rng, value_rng = random.split(self.rng, 3)
    self.act_rng = self.rng
    for rng in random.split(policy_rng, self.n_groups):
      self.params.policies.append(policy_init(
        rng, data.obs, data.state_reset, data.state, 
        prev_info, data.action_mask
      ))
    
    self.params.vs = []
    value_init, self.modules.value = self.build_net(
      name='value', return_init=True)
    for rng in random.split(value_rng, self.n_groups):
      self.params.vs.append(value_init(
        rng, data.global_state, data.state_reset, data.state, prev_info
      ))
    self._init_rnn()

  def _init_rnn(self):
    if self.policy_rnn_init:
      for policy_params in self.params.policies:
        for k in policy_params:
          if 'gru' in k or 'lstm' in k:
            for kk, vv in policy_params[k].items():
              if not kk.endswith('b'):
                self.rng, rng = random.split(self.rng)
                policy_params[k][kk] = reset_weights(vv, rng, self.policy_rnn_init)
    if self.value_rnn_init:
      for value_params in self.params.vs:
        for k in value_params:
          if 'gru' in k or 'lstm' in k:
            for kk, vv in value_params[k].items():
              if not kk.endswith('b'):
                self.rng, rng = random.split(self.rng)
                value_params[k][kk] = reset_weights(vv, rng, self.value_rnn_init)

  def compile_model(self):
    self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))
    self.jit_action_logprob = jax.jit(self.action_logprob)
    self.jit_value = jax.jit(self.raw_value)

  def action(self, data, evaluation):
    for d in data:
      if 'global_state' not in d:
        d.global_state = d.obs
    action, stats, state = super().action(data, evaluation)
    return action, stats, state

  def raw_action(self, params, rng, data, evaluation=False):
    rngs = random.split(rng, self.n_groups)
    all_actions = []
    all_stats = []
    all_states = []
    for gid, (p, v, rng) in enumerate(zip(params.policies, params.vs, rngs)):
      agent_rngs = random.split(rng, 3)
      d = data[gid]
      state = d.pop('state', AttrDict())
      if self.has_rnn:
        d = jax.tree_map(lambda x: jnp.expand_dims(x, 1) , d)
      act_outs, state.policy = self.forward_policy(p, agent_rngs[0], d, state.policy)
      act_dists = self.policy_dist(act_outs, evaluation)

      if evaluation:
        action = dict2AttrDict({k: ad.sample(seed=agent_rngs[1]) for k, ad in act_dists.items()})
        stats = AttrDict()
      else:
        if len(act_dists) == 1:
          action, logprob = act_dists[DEFAULT_ACTION].sample_and_log_prob(seed=agent_rngs[1])
          action = dict2AttrDict({DEFAULT_ACTION: action})
          stats = act_dists[DEFAULT_ACTION].get_stats('mu')
          stats = dict2AttrDict(stats)
          stats.mu_logprob = logprob
        else:
          action = AttrDict()
          logprob = AttrDict()
          stats = AttrDict(mu_logprob=0)
          for k, ad in act_dists.items():
            a, lp = ad.sample_and_log_prob(seed=agent_rngs[1])
            action[k] = a
            logprob[k] = lp
            stats.update(ad.get_stats(f'{k}_mu'))
            stats.mu_logprob = stats.mu_logprob + lp
        value, state.value = self.forward_value(
          v, agent_rngs[2], d, state=state.value, return_state=True
        )
        stats.value = value
      if self.has_rnn:
        # squeeze the sequential dimension
        action, stats = jax.tree_map(
          lambda x: jnp.squeeze(x, 1), (action, stats))
        all_states.append(state)
      else:
        all_states = None

      all_actions.append(action)
      all_stats.append(stats)
    
    action = batch_dicts(all_actions, concat_along_unit_dim)
    stats = batch_dicts(all_stats, func=concat_along_unit_dim)

    return action, stats, all_states

  def action_logprob(self, params, rng, data):
    data.state_reset, _ = jax_utils.split_data(
      data.state_reset, axis=1)
    if 'state' in data:
      data.state = tree_slice(data.state, indices=0, axis=1)
    state = data.pop('state', AttrDict())
    data.action_mask = get_action_mask(data.action)
    act_out = self.forward_policy(params, rng, data, state=state.policy, return_state=False)
    act_dists = self.policy_dist(act_out)
    if len(act_dists) == 1:
      logprob = act_dists[DEFAULT_ACTION].log_prob(data.action[DEFAULT_ACTION])
    else:
      logprob = sum([act_dists[k].log_prob(data.action[k]) for k in act_dists])

    return logprob

  def raw_value(self, params, rng, data):
    vs = []
    for p, d in zip(params, data):
      state = d.pop('state', AttrDict())
      if self.has_rnn:
        d = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1) , d)
      v = self.forward_value(p, rng, d, state.value, return_state=False)
      vs.append(v)
    vs = jnp.concatenate(vs, -1)
    if self.has_rnn:
      vs = jnp.squeeze(vs, 1)
    return vs

  def compute_value(self, data):
    self.act_rng, rng = random.split(self.act_rng)
    value = self.jit_value(self.params.vs, rng, data)
    return value

  """ RNN Operators """
  def get_initial_state(self, batch_size, name='default'):
    name = f'{name}_{batch_size}'
    if name in self._initial_states:
      return self._initial_states[name]
    if not self.has_rnn:
      return None
    data = construct_fake_data(self.env_stats, self.aid, batch_size=batch_size)
    prev_info = jnp.concatenate([v for v in data.prev_info.values()], -1) \
      if self.config.use_prev_info and data.prev_info else None
    action_mask = AttrDict()
    for k, v in data.action.items():
      if k.endswith('_mask'):
        action_mask[k.replace('_mask', '')] = v
    states = []
    for uids, p, v in zip(self.gid2uids, self.params.policies, self.params.vs):
      state = AttrDict()
      d = data.slice(indices=uids, axis=TRAIN_AXIS.UNIT)
      am = action_mask.slice(indices=uids, axis=TRAIN_AXIS.UNIT)
      pi = prev_info.slice(indices=uids, axis=TRAIN_AXIS.UNIT) \
        if prev_info is not None else None
      _, state.policy = self.modules.policy(
        p, self.act_rng, d.obs, reset=d.state_reset, 
        prev_info=pi, action_mask=am)
      _, state.value = self.modules.value(
        v, self.act_rng, d.global_state, 
        reset=d.state_reset, prev_info=pi)
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
