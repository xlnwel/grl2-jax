import numpy as np
from typing import Dict
from jax import tree_map
import torch
import torch.nn as nn

from th.core.names import DEFAULT_ACTION
from th.core.typing import dict2AttrDict
from th.core.utils import tpdv
from th.nn.func import nn_registry
from th.nn.mlp import MLP
from th.nn.utils import get_activation, init_linear
""" Source this file to register Networks """


class CategoricalOutput(nn.Module):
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


class MultivariateNormalOutput(nn.Module):
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
    device='cpu', 
    **config
  ):
    super().__init__()
    self.tpdv = tpdv(device)
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
    self.net = MLP(input_dim, **self.config)
    self.heads: Dict[str, nn.Module] = {}
    for k in action_dim:
      if is_action_discrete[k]:
        self.heads[k] = CategoricalOutput(
          self.config.rnn_units, action_dim[k], 
          out_w_init, out_b_init, out_scale)
      else:
        self.heads[k] = MultivariateNormalOutput(
          self.config.rnn_units, action_dim[k], 
          out_w_init, out_b_init, out_scale, 
          sigmoid_scale=sigmoid_scale, std_x_coef=std_x_coef, 
          std_y_coef=std_y_coef, init_std=init_std)
    for k, v in self.heads.items():
      setattr(self, f'head_{k}', v)

  def forward(self, x, reset=None, state=None, action_mask=None):
    x, reset, state, action_mask = tree_map(
      lambda x: x.to(**self.tpdv), (x, reset, state, action_mask))
    if self.use_feature_norm:
      x = self.pre_ln(x)
    x = self.net(x, reset, state)
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
    device='cpu', 
    **config
  ):
    super().__init__()
    self.tpdv = tpdv(device)
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
    x, reset, state = tree_map(
      lambda x: x.to(**self.tpdv), (x, reset, state))
    if self.use_feature_norm:
      x = self.pre_ln(x)
    x = self.net(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x

    if x.shape[-1] == 1:
      x = x.squeeze(-1)
    value = self.out_act(x)

    return value, state


@nn_registry.register('vnorm')
class ValueNorm(nn.Module):
  """ Normalize a vector of observations - across the first norm_axes dimensions"""

  def __init__(
    self, 
    input_shape, 
    axis=(0, 1), 
    beta=0.99999, 
    per_element_update=False, 
    epsilon=1e-5, 
    device='cpu'
  ):
    super(ValueNorm, self).__init__()
    self.tpdv = tpdv(device)

    if isinstance(axis, int):
      axis = (axis, )
    elif isinstance(axis, (tuple, list)):
      axis = tuple(axis)
    elif axis is None:
      pass
    else:
      raise ValueError(f'Invalid axis({axis}) of type({type(axis)})')

    if isinstance(axis, tuple):
      assert axis == tuple(range(len(axis))), \
        f'Axis should only specifies leading axes so that '\
        f'mean and var can be broadcasted automatically when normalizing. '\
        f'But receving axis = {axis}'
  
    self.input_shape = input_shape
    self.axis = axis
    self.epsilon = epsilon
    self.beta = beta
    self.per_element_update = per_element_update

    self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
    self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
    self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)
    
    self.reset_parameters()

  def reset_parameters(self):
    self.running_mean.zero_()
    self.running_mean_sq.zero_()
    self.debiasing_term.zero_()

  def running_mean_var(self):
    debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
    debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
    debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
    return debiased_mean, debiased_var

  @torch.no_grad()
  def update(self, input_vector):
    if type(input_vector) == np.ndarray:
      input_vector = torch.from_numpy(input_vector)
    input_vector = input_vector.to(**self.tpdv)

    batch_mean = input_vector.mean(dim=self.axis)
    batch_sq_mean = (input_vector ** 2).mean(dim=self.axis)

    if self.per_element_update:
      batch_size = np.prod(input_vector.size()[self.axis])
      weight = self.beta ** batch_size
    else:
      weight = self.beta

    self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
    self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
    self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

  def normalize(self, input_vector):
    # Make sure input is float32
    if type(input_vector) == np.ndarray:
      input_vector = torch.from_numpy(input_vector)
    input_vector = input_vector.to(**self.tpdv)

    mean, var = self.running_mean_var()
    out = (input_vector - mean) / torch.sqrt(var)
    
    return out

  def denormalize(self, input_vector):
    """ Transform normalized data back into original distribution """
    if type(input_vector) == np.ndarray:
      input_vector = torch.from_numpy(input_vector)
    input_vector = input_vector.to(**self.tpdv)

    mean, var = self.running_mean_var()
    out = input_vector * torch.sqrt(var) + mean
    
    return out


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
  x, state = policy(x, reset)
  for k, v in x.items():
    print('output', k, v.shape)
  print('state', state.shape)
