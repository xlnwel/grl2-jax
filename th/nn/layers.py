import collections
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from th.nn.utils import get_initializer


LSTMState = collections.namedtuple('LSTMState', 'h c')


class LSTM(nn.Module):
  def __init__(self, input_dim, hidden_size):
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.linear = nn.Linear(input_dim + hidden_size, 4 * hidden_size)
  
  def forward(self, x, state, reset):
    state = tree_map(lambda s: s * (1 - reset)[..., None], state)
    xh = torch.concat((x, state[0][0]), dim=-1)
    gated = self.linear(xh)
    i, g, f, o = torch.split(gated, (self.hidden_size, ) * 4, -1)
    f = torch.sigmoid(f + 1)
    c = f * state[1][0] + torch.sigmoid(i) * torch.tanh(g)
    h = torch.sigmoid(o) * torch.tanh(c)
    return h, (h[None], c[None])


# class GRU(nn.Module):
#   def __init__(self, input_dim, hidden_size):
#     super(GRU, self).__init__()
#     self.hidden_size = hidden_size
#     self.linear = nn.Linear(input_dim + hidden_size, 3 * hidden_size)
  
#   def forward(self, x, state, reset):
#     state = tree_map(lambda s: s * (1 - reset)[..., None], state)
#     xh = torch.concat((x, state[0]), dim=1)

"""RNN modules."""
class RNNLayer(nn.Module):
  def __init__(self, inputs_dim, outputs_dim, rnn_type, rnn_layers=1, rnn_init='orthogonal', rnn_norm=False):
    super(RNNLayer, self).__init__()
    self.rnn_type = rnn_type
    self._rnn_layers = rnn_layers

    if rnn_type == 'lstm':
      self.rnn = nn.LSTM(inputs_dim, outputs_dim, num_layers=self._rnn_layers)
    elif rnn_type == 'gru':
      self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._rnn_layers)
    else:
      raise NotImplementedError(rnn_type)
    for name, param in self.rnn.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0)
      elif 'weight' in name:
        w_init = get_initializer(rnn_init)
        w_init(param)
    if rnn_norm:
      self.norm = nn.LayerNorm(outputs_dim)
    else:
      self.norm = None

  def forward(self, x, state, reset):
    # outputs = []
    # for i in range(x.size(0)):
    #   mask = 1 - reset[i].unsqueeze(-1).contiguous()
    #   state = tree_map(lambda x: x * mask, state)
    #   h, state = self.rnn(x[i].unsqueeze(0), state)
    #   outputs.append(h)
    # Let's figure out which steps in the sequence have a zero for any agent
    # We will always assume t=0 has a zero in it as that makes the logic cleaner
    is_reset = ((reset[1:] == 1.0)
                .any(dim=-1)
                .nonzero()
                .squeeze()
                .cpu())

    # +1 to correct the reset[1:]
    if is_reset.dim() == 0:
      # Deal with scalar
      is_reset = [is_reset.item() + 1]
    else:
      is_reset = (is_reset + 1).numpy().tolist()

    # # add t=0 and t=T to the list
    is_reset = [0] + is_reset + [x.size(0)]

    outputs = []
    for i in range(len(is_reset) - 1):
      # We can now process steps that don't have any zeros in masks together!
      # This is much faster
      start_idx = is_reset[i]
      end_idx = is_reset[i + 1]
      mask = 1 - reset[start_idx].unsqueeze(-1).contiguous()
      state = tree_map(lambda x: (x * mask).contiguous(), state)
      h, state = self.rnn(x[start_idx:end_idx], state)
      outputs.append(h)

    # assert len(outputs) == T
    # x is a (T, N, -1) tensor
    x = torch.cat(outputs, dim=0)

    if self.norm:
      x = self.norm(x)
    if self.rnn_type == 'lstm':
      state = LSTMState(*state)

    return x, state
