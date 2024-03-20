import numpy as np
from typing import Dict
import torch
import torch.nn as nn

from core.names import DEFAULT_ACTION
from core.typing import dict2AttrDict
from nn.func import nn_registry
from th.nn.mlp import MLP
from th.nn.utils import get_activation, init_linear
""" Source this file to register Networks """



class Categorical(nn.Module):
  def __init__(
    self, 
    num_inputs, 
    num_outputs, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=0.01
  ):
    super().__init__()
    self.linear = nn.Linear(num_inputs, num_outputs)
    init_linear(self.linear, out_w_init, out_b_init, out_scale)

  def forward(self, x, action_mask=None):
    x = self.linear(x)
    if action_mask is not None:
      x[action_mask == 0] = -1e10
    return x
    # return torch.distributions.Categorical(logits=x)


class DiagGaussian(nn.Module):
  def __init__(
    self, 
    num_inputs, 
    num_outputs, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=0.01, 
    sigmoid_scale=True, 
    std_x_coef=1., 
    std_y_coef=.5, 
    init_std=.2
  ):
    super().__init__()
    self.linear = nn.Linear(num_inputs, num_outputs)
    init_linear(self.linear, out_w_init, out_b_init, out_scale)
    self.sigmoid_scale = sigmoid_scale
    self.std_x_coef = std_x_coef
    self.std_y_coef = std_y_coef
    if sigmoid_scale:
      self.logstd = nn.Parameter(std_x_coef + torch.zeros(num_outputs))
    else:
      self.logstd = nn.Parameter(np.log(init_std) + torch.zeros(num_outputs))

  def forward(self, x):
    mean = self.linear(x)
    if self.sigmoid_scale:
      scale = torch.sigmoid(self.logstd / self.std_x_coef) * self.std_y_coef
    else:
      scale = torch.exp(self.logstd)
    return mean, scale
    # return torch.distributions.Normal(mean, scale)


@nn_registry.register('policy')
class Policy(nn.Module):
  def __init__(
    self, 
    input_dim, 
    is_action_discrete, 
    action_dim, 
    out_act=None, 
    init_std=.2, 
    sigmoid_scale=True, 
    std_x_coef=1., 
    std_y_coef=.5, 
    use_action_mask={DEFAULT_ACTION: False}, 
    use_feature_norm=False, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=.01, 
    **config
  ):
    super().__init__()
    self.config = dict2AttrDict(config, to_copy=True)
    self.action_dim = action_dim
    self.is_action_discrete = is_action_discrete

    self.out_act = out_act
    self.init_std = init_std
    self.sigmoid_scale = sigmoid_scale
    self.std_x_coef = std_x_coef
    self.std_y_coef = std_y_coef
    self.use_action_mask = use_action_mask
    self.use_feature_norm = use_feature_norm
    if self.use_feature_norm:
      self.pre_ln = nn.LayerNorm(input_dim)
    self.mlp = MLP(input_dim, **self.config)
    self.heads: Dict[str, nn.Module] = {}
    for k in action_dim:
      if is_action_discrete[k]:
        self.heads[k] = Categorical(
          self.config.rnn_units, action_dim[k], 
          out_w_init, out_b_init, out_scale)
      else:
        self.heads[k] = DiagGaussian(
          self.config.rnn_units, action_dim[k], 
          out_w_init, out_b_init, out_scale, 
          sigmoid_scale=sigmoid_scale, std_x_coef=std_x_coef, 
          std_y_coef=std_y_coef, init_std=init_std)
    for k, v in self.heads.items():
      setattr(self, f'head_{k}', v)

  def forward(self, x, reset=None, state=None, action_mask=None):
    if self.use_feature_norm:
      x = self.pre_ln(x)
    x = self.mlp(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    
    outs = {}
    for name, layer in self.heads.items():
      if self.is_action_discrete[name]:
        am = action_mask[name] if action_mask is not None else None
        d = layer(x, action_mask=am)
      else:
        d = layer(x)
      outs[name] = d
    return outs, state


@nn_registry.register('value')
class Value(nn.Module):
  def __init__(
    self, 
    input_dim, 
    out_act=None, 
    out_size=1, 
    use_feature_norm=False, 
    **config
  ):
    super().__init__()
    self.config = dict2AttrDict(config, to_copy=True)
    self.out_act = get_activation(out_act)
    self.out_size = out_size
    self.use_feature_norm = use_feature_norm
    if self.use_feature_norm:
      self.pre_ln = nn.LayerNorm(input_dim)
    self.net = MLP(
      input_dim, 
      **self.config, 
      out_size=self.out_size, 
    )

  def __call__(self, x, reset=None, state=None):
    if self.use_feature_norm:
      x = self.pre_ln(x)
    x = self.mlp(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x

    if x.shape[-1] == 1:
      x = x.squeeze(-1)
    value = self.out_act(x)

    return value, state


if __name__ == '__main__':
  b = 4
  s = 3
  u = 2
  d = 5
  is_action_discrete = {'action_disc': True, 'action_cont': False}
  action_dim = {'action_disc': 4, 'action_cont': 3}
  use_action_mask={'action_disc': False, 'action_cont': False}
  config = dict(
    input_dim=d, 
    is_action_discrete=is_action_discrete, 
    action_dim=action_dim, 
    use_action_mask=use_action_mask, 
    units_list=[64, 64],
    w_init='orthogonal',
    activation='relu', 
    norm='layer',
    norm_after_activation=True,
    out_scale=.01,
    rnn_type='lstm',
    rnn_units=64,
    rnn_init=None,
    rnn_norm='layer',
  )

  policy = Policy(**config)
  print(policy)
  x = torch.rand(b, s, u, d)
  reset = torch.randn(b, s, u) < .5
  # state = torch.zeros(1, b, u, d)
  x, state = policy(x, reset, return_action=True)
  for k, v in x.items():
    print('output', k, v.shape)
  print('state', state.shape)
