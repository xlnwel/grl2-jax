import os
import torch
from torch.utils._pytree import tree_map

from envs.utils import get_action_mask
from th.core.names import DEFAULT_ACTION
from th.core.typing import AttrDict, dict2AttrDict
from th.tools.th_utils import to_tensor
from th.algo.ma_common.elements.model import MAModelBase, \
  setup_config_from_envstats, construct_fake_data
from tools.file import source_file


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class Model(MAModelBase):
  def build_nets(self):
    aid = self.config.get('aid', 0)
    data = construct_fake_data(self.env_stats, aid=aid)

    self.policy = self.build_net(data.obs.shape[-1], name='policy')
    self.value = self.build_net(data.global_state.shape[-1], name='value')
    self.vnorm = self.build_net(len(self.env_stats.aid2uids[aid]), name='vnorm')

  @property
  def theta(self):
    return AttrDict(
      policy=self.policy.parameters(),
      value=self.value.parameters()
    )

  def action(self, data, evaluation):
    if 'global_state' not in data:
      data.global_state = data.obs
    action, stats, state = super().action(data, evaluation)
    return action, stats, state

  @torch.no_grad()
  def raw_action(self, data, evaluation=False):
    state = data.pop('state', AttrDict())
    # add the sequential dimension
    if self.has_rnn:
      data = tree_map(lambda x: torch.unsqueeze(x, 1), data)
    act_outs, state.policy = self.forward_policy(data, state.policy)
    act_dists = self.policy_dist(act_outs, evaluation)

    if evaluation:
      action = {k: ad.sample() for k, ad in act_dists.items()}
      stats = AttrDict()
    else:
      if len(act_dists) == 1:
        action = act_dists[DEFAULT_ACTION].sample()
        logprob = act_dists[DEFAULT_ACTION].log_prob(action)
        action = dict2AttrDict({DEFAULT_ACTION: action})
        stats = act_dists[DEFAULT_ACTION].get_stats('mu')
        stats = dict2AttrDict(stats)
        stats.mu_logprob = logprob
      else:
        action = AttrDict()
        logprob = AttrDict()
        stats = AttrDict(mu_logprob=0)
        for k, ad in act_dists.items():
          a = ad.sample()
          lp = ad.log_prob(a)
          action[k] = a
          logprob[k] = lp
          k = k.replace('action_', '')
          stats.update(ad.get_stats(f'{k}_mu'))
          stats.mu_logprob = stats.mu_logprob + lp
        
      value, state.value = self.forward_value(data, state.value)
      stats.value = value
    if self.has_rnn:
      # squeeze the sequential dimension
      action, stats = tree_map(
        lambda x: torch.squeeze(x, 1), (action, stats))
    if state.policy is None and state.value is None:
      state = None
    else:
      state = tree_map(lambda x: x.cpu(), state)
    
    return action, stats, state

  @torch.no_grad()
  def compute_value(self, data):
    state = data.pop('state', AttrDict())
    data = to_tensor(data, self.tpdv)
    if self.has_rnn:
      data = tree_map(lambda x: torch.unsqueeze(x, 1) , data)
    value, _ = self.forward_value(data, state.value)
    if self.has_rnn:
      value = torch.squeeze(value, 1)
    value = value.cpu().numpy()
    return value

  """ RNN Operators """
  @torch.no_grad()
  def get_initial_state(self, batch_size, name='default'):
    name = f'{name}_{batch_size}'
    if name in self._initial_states:
      return self._initial_states[name]
    if not self.has_rnn:
      return None
    data = construct_fake_data(self.env_stats, self.aid, batch_size=batch_size)
    data.action_mask = get_action_mask(data.action)
    state = AttrDict()
    _, state.policy = self.forward_policy(data)
    _, state.value = self.forward_value(data)
    self._initial_states[name] = tree_map(lambda x: torch.zeros_like(x).cpu(), state)

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


if __name__ == '__main__':
  from tools.yaml_op import load_config
  from envs.func import create_env
  from th.core.utils import set_seed
  set_seed(50)
  config = load_config('th.algo/ppo/configs/template')
  
  env = create_env(config.env)
  config.model.root_dir = 'debug/root'
  config.model.model_name = 'a0'
  model_config = config.model.copy()
  model_config.policy.units_list = [2, 2]
  model_config.policy.rnn_units = 2
  model_config.value.units_list = [2, 2]
  model_config.value.rnn_units = 2
  model = create_model(config.model, env.stats())
  data = construct_fake_data(env.stats(), 0)
  data.pop('action')
  data = tree_map(lambda x: torch.squeeze(x, 1), data)
  action, stats, state = model.action(data, False)
  print('action', action)
  print('state', state)
