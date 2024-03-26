import torch
import torch.nn as nn
from torch.utils._pytree import tree_map


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
  def __init__(self, inputs_dim, outputs_dim, rnn_type, rnn_layers=1, rnn_norm=False):
    super(RNNLayer, self).__init__()
    self.rnn_type = rnn_type
    self._rnn_layers = rnn_layers

    if rnn_type == 'lstm':
      self.rnn = LSTM(inputs_dim, outputs_dim)
    elif rnn_type == 'gru':
      self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._rnn_layers)
    else:
      raise NotImplementedError(rnn_type)
    for name, param in self.rnn.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0)
      elif 'weight' in name:
        nn.init.orthogonal_(param)
    if rnn_norm:
      self.norm = nn.LayerNorm(outputs_dim)
    else:
      self.norm = None

  def forward(self, x, state, reset):
    outputs = []
    for i in range(x.size(0)):
      h, state = self.rnn(x[i], state, reset[i])
      outputs.append(h)
    # # Let's figure out which steps in the sequence have a zero for any agent
    # # We will always assume t=0 has a zero in it as that makes the logic cleaner
    # has_zeros = ((masks[1:] == 0.0)
    #         .any(dim=-1)
    #         .nonzero()
    #         .squeeze()
    #         .cpu())

    # # +1 to correct the masks[1:]
    # if has_zeros.dim() == 0:
    #   # Deal with scalar
    #   has_zeros = [has_zeros.item() + 1]
    # else:
    #   has_zeros = (has_zeros + 1).numpy().tolist()

    # # add t=0 and t=T to the list
    # has_zeros = [0] + has_zeros + [x.size(0)]

    # outputs = []
    # for i in range(len(has_zeros) - 1):
    #   # We can now process steps that don't have any zeros in masks together!
    #   # This is much faster
    #   start_idx = has_zeros[i]
    #   end_idx = has_zeros[i + 1]
    #   temp = tree_map(lambda x: x * masks[start_idx].view(1, -1, 1).repeat(self._rnn_layers, 1, 1).contiguous(), state)
    #   temp = (temp[0], torch.zeros_like(temp[1]))
    #   # temp = (torch.zeros_like(temp[0]), temp[1])
    #   print(i, 'torch input state', temp)
    #   rnn_scores, state = self.rnn(x[start_idx:end_idx], temp)
    #   print(i, 'torch output state', state)
    #   outputs.append(rnn_scores)

    # assert len(outputs) == T
    # x is a (T, N, -1) tensor
    x = torch.cat(outputs, dim=0)

    if self.norm:
      x = self.norm(x)
    return x, state
