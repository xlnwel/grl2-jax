import os
import logging
import jax
from jax import random
import jax.numpy as jnp

from core.typing import AttrDict
from tools.file import source_file
from tools.utils import batch_dicts
from algo.ma_common.elements.model import *


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


class Model(MAModelBase):
  def build_nets(self):
    aid = self.config.get('aid', 0)
    data = construct_fake_data(self.env_stats, aid=aid)

    self.params.policy, self.modules.policy = self.build_net(
      data.obs, data.state_reset, data.state, data.action_mask, name='policy')
    self.params.value, self.modules.value = self.build_net(
      data.global_state, data.state_reset, data.state, name='value')

  def compile_model(self):
    self.jit_action = jax.jit(
      self.raw_action, static_argnames=('evaluation'))

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
    act_out, state = self.forward_policy(params.policy, rngs[0], data, state)
    act_dist = self.policy_dist(act_out, evaluation)

    if evaluation:
      action = act_dist.mode()
      stats = AttrDict()
    else:
      stats = act_dist.get_stats('mu')
      action, logprob = act_dist.sample_and_log_prob(seed=rngs[1])
      value, state.value = self.modules.value(
        params.value, 
        rngs[2], 
        data.global_state, 
        data.state_reset, 
        state.value
      )
      stats.update({'mu_logprob': logprob, 'value': value})
    if self.has_rnn:
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
    data = construct_fake_data(self.env_stats, batch_size)
    data = batch_dicts(data, lambda x: jnp.concatenate(x, axis=2))
    state = AttrDict()
    _, state.policy = self.modules.policy(
      self.params.policy, 
      self.act_rng, 
      data.obs, 
      data.state_reset
    )
    _, state.value = self.modules.value(
      self.params.value, 
      self.act_rng, 
      data.global_state, 
      data.state_reset
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
