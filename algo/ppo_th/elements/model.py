import os
import jax
from jax import random
import jax.numpy as jnp

from env.utils import get_action_mask
from core.names import DEFAULT_ACTION
from core.typing import AttrDict, dict2AttrDict
from th.core.elements.model import Model as ModelBase
from tools.file import source_file
from tools.utils import batch_dicts

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


def construct_fake_data(env_stats, aid, batch_size=1):
  shapes = env_stats.obs_shape[aid]
  dtypes = env_stats.obs_dtype[aid]
  action_dim = env_stats.action_dim[aid]
  action_dtype = env_stats.action_dtype[aid]
  use_action_mask = env_stats.use_action_mask[aid]
  n_units = len(env_stats.aid2uids[aid])
  basic_shape = (batch_size, 1, n_units)
  data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
    for k, v in shapes.items()}
  data = dict2AttrDict(data)
  data.setdefault('global_state', data.obs)
  data.action = AttrDict()
  for k in action_dim.keys():
    data.action[k] = jnp.zeros((*basic_shape, action_dim[k]), action_dtype[k])
    if use_action_mask[k]:
      data.action[f'{k}_mask'] = jnp.ones((*basic_shape, action_dim[k]), action_dtype[k])
  data.state_reset = jnp.zeros(basic_shape, jnp.float32)
  return data


class Model(ModelBase):
  def build_nets(self):
    aid = self.config.get('aid', 0)
    data = construct_fake_data(self.env_stats, aid=aid)

    self.policy = self.build_net(data.obs.shape[-1], name='policy')
    self.value = self.build_net(data.global_state.shape[-1], name='value')

  def action(self, data, evaluation):
    if 'global_state' not in data:
      data.global_state = data.obs
    return super().action(data, evaluation)

  def raw_action(self, data, evaluation=False):
    state = data.pop('state', AttrDict())
    # add the sequential dimension
    if self.has_rnn:
      data = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), data)
    act_outs, state.policy = self.policy(data.obs, data.state_reset, state.policy)
    act_outs, state = self.forward_policy(params.policy, rngs[0], data, state)
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
        
      value, state.value = self.modules.value(
        params.value, 
        rngs[2], 
        data.global_state, 
        data.state_reset, 
        state.value
      )
      stats['value'] = value
    if self.has_rnn:
      # squeeze the sequential dimension
      action, stats = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, 1), (action, stats))
    if state.policy is None and state.value is None:
      state = None
    
    return action, stats, state

  def compute_value(self, data):
    @jax.jit
    def comp_value(params, rng, data):
      state = data.pop('state', AttrDict())
      if self.has_rnn:
        data = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1) , data)
      v, _ = self.modules.value(
        params, 
        rng, 
        data.global_state, 
        data.state_reset, 
        state.value
      )
      if self.has_rnn:
        v = jnp.squeeze(v, 1)
      return v
    self.act_rng, rng = random.split(self.act_rng)
    value = comp_value(self.params.value, rng, data)
    return value

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
    _, state.value = self.modules.value(
      self.params.value, 
      self.act_rng, 
      data.global_state, 
      reset=data.state_reset
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
