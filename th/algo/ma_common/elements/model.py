import numpy as np
import torch

from tools.utils import tree_map
from th.core.elements.model import Model as ModelCore
from th.core.typing import AttrDict, dict2AttrDict
from th.nn.layers import LSTMState
from th.tools import th_dist


def construct_fake_data(env_stats, aid, batch_size=1):
  shapes = env_stats.obs_shape[aid]
  dtypes = env_stats.obs_dtype[aid]
  action_dim = env_stats.action_dim[aid]
  action_shape = env_stats.action_shape[aid]
  action_dtype = env_stats.action_dtype[aid]
  use_action_mask = env_stats.use_action_mask[aid]
  n_units = len(env_stats.aid2uids[aid])
  basic_shape = (batch_size, 1, n_units)
  data = {k: np.zeros((*basic_shape, *v), dtypes[k]) 
    for k, v in shapes.items()}
  data = dict2AttrDict(data)
  data.setdefault('global_state', data.obs)
  data.action = AttrDict()
  data.prev_info = AttrDict()
  for k in action_dim.keys():
    data.action[k] = np.zeros((*basic_shape, *action_shape[k]), action_dtype[k])
    data.prev_info[k] = np.zeros((*basic_shape, action_dim[k]), action_dtype[k])
    if use_action_mask[k]:
      data.action[f'{k}_mask'] = np.ones((*basic_shape, action_dim[k]), action_dtype[k])
  data.state_reset = np.zeros(basic_shape, np.float32)
  data = tree_map(lambda x: torch.from_numpy(x), data)

  return data


def setup_config_from_envstats(config, env_stats):
  idx = config.gid or config.aid
  config.policy.action_dim = env_stats.action_dim[idx]
  config.policy.is_action_discrete = env_stats.is_action_discrete[idx]
  config.policy.use_action_mask = env_stats.use_action_mask[idx]

  return config


class MAModelBase(ModelCore):
  def forward_policy(self, data, state=None, return_state=True):
    act_outs, state = self.modules.policy(
      data.obs, data.state_reset, state, 
      action_mask=data.action_mask, 
    )

    if return_state:
      return act_outs, state
    return act_outs

  def policy_dist(self, act_outs, evaluation=False):
    dists = {}
    for k, v in act_outs.items():
      if self.is_action_discrete[k]:
        if evaluation and self.config.get('eval_act_temp', 0) > 0:
          v = v / self.config.eval_act_temp
        dists[k] = th_dist.Categorical(logits=v)
      else:
        loc, scale = v
        if evaluation and self.config.get('eval_act_temp', 0) > 0:
          scale = scale * self.config.eval_act_temp
        dists[k] = th_dist.MultivariateNormalDiag(loc, scale)
    return dists
  
  def forward_value(self, data, state=None, return_state=True):
    value, state = self.modules.value(
      data.global_state, data.state_reset, state
    )

    if return_state:
      return value, state
    return value  

  """ RNN Operators """
  @property
  def has_rnn(self):
    has_rnn = False
    for v in self.config.values():
      if isinstance(v, dict):
        has_rnn = v.rnn_type is not None
      if has_rnn:
        break
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
      'lstm': LSTMState._fields, 
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
      'lstm': LSTMState, 
      'gru': None, 
      None: None
    }
    state_type = dict2AttrDict({
      k: type_map[v.rnn_type] 
      for k, v in self.config.items() if isinstance(v, dict)
    })
    return state_type
