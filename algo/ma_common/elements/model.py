from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk

from core.elements.model import Model as ModelCore
from core.typing import AttrDict, dict2AttrDict, tree_slice
from jax_tools import jax_dist


def construct_fake_data(env_stats, batch_size=1, aid=None):
  if aid is None:
    all_data = []
    for i, uids in enumerate(env_stats.aid2uids):
      shapes = env_stats.obs_shape[i]
      dtypes = env_stats.obs_dtype[i]
      action_dim = env_stats.action_dim[i]
      action_dtype = env_stats.action_dtype[i]
      basic_shape = (batch_size, 1, len(env_stats.aid2uids[i]))
      data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
        for k, v in shapes.items()}
      data = dict2AttrDict(data)
      data.setdefault('global_state', data.obs)
      data.action = jnp.zeros((*basic_shape, action_dim), action_dtype)
      data.joint_action = jnp.zeros((*basic_shape, env_stats.n_units*action_dim), action_dtype)
      data.state_reset = jnp.zeros(basic_shape, jnp.float32)
      all_data.append(data)
    return all_data
  else:
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    action_dtype = env_stats.action_dtype[aid]
    basic_shape = (batch_size, 1, len(env_stats.aid2uids[aid]))
    data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
      for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.action = jnp.zeros((*basic_shape, action_dim), action_dtype)
    data.joint_action = jnp.zeros((*basic_shape[:2], 1, env_stats.n_units*action_dim), action_dtype)
    data.state_reset = jnp.zeros(basic_shape, jnp.float32)
    return data


def setup_config_from_envstats(config, env_stats):
  if 'aid' in config:
    aid = config['aid']
    config.policy.action_dim = env_stats.action_dim[aid]
    config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
  else:
    config.policy.action_dim = env_stats.action_dim
    config.policy.is_action_discrete = env_stats.is_action_discrete
    config.policy.action_low = env_stats.action_low
    config.policy.action_high = env_stats.action_high

  return config


class MAModelBase(ModelCore):
  def add_attributes(self):
    self.n_agents = self.env_stats.n_agents
    self.aid2uids = self.env_stats.aid2uids
    aid = self.config.get('aid', 0)
    self.is_action_discrete = self.env_stats.is_action_discrete[aid]

  def forward_policy(self, params, rng, data, state=AttrDict(), return_state=True):
    act_out, state.policy = self.modules.policy(
      params, 
      rng, 
      data.obs, 
      data.state_reset, 
      state.policy, 
      action_mask=data.action_mask, 
    )

    if return_state:
      return act_out, state
    return act_out

  def policy_dist(self, act_out, evaluation=False):
    if self.is_action_discrete:
      if evaluation and self.config.get('eval_act_temp', 0) > 0:
        act_out = act_out / self.config.eval_act_temp
      dist = jax_dist.Categorical(logits=act_out)
    else:
      loc, scale = act_out
      if evaluation and self.config.get('eval_act_temp', 0) > 0:
        scale = scale * self.config.eval_act_temp
      dist = jax_dist.MultivariateNormalDiag(
        loc, scale, joint_log_prob=self.config.joint_log_prob)

    return dist

  """ RNN Operators """
  @property
  def has_rnn(self):
    has_rnn = False
    for v in self.config.values():
      if isinstance(v, dict):
        has_rnn = v.rnn_type is not None
    return has_rnn

  @property
  def state_size(self):
    if not self.has_rnn:
      return None
    state_size = dict2AttrDict({
      k: v.rnn_units 
      for k, v in self.config.items() if isinstance(v, dict)
    })
    return state_size
  
  @property
  def state_keys(self):
    if not self.has_rnn:
      return None
    key_map = {
      'lstm': hk.LSTMState._fields, 
      'gru': None, 
      None: None
    }
    state_keys = dict2AttrDict({
      k: key_map[v.rnn_type] 
      for k, v in self.config.items() if isinstance(v, dict)
    })
    return state_keys

  @property
  def state_type(self):
    if not self.has_rnn:
      return None
    type_map = {
      'lstm': hk.LSTMState, 
      'gru': None, 
      None: None
    }
    state_type = dict2AttrDict({
      k: type_map[v.rnn_type] 
      for k, v in self.config.items() if isinstance(v, dict)
    })
    return state_type
