import torch
import torch.nn as nn

"""RNN modules."""


class RNNLayer(nn.Module):
  def __init__(self, inputs_dim, outputs_dim, rnn_layers=1, rnn_norm=False):
    super(RNNLayer, self).__init__()
    self._rnn_layers = rnn_layers

    self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._rnn_layers)
    for name, param in self.rnn.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0)
      elif 'weight' in name:
        nn.init.orthogonal_(param)
    if rnn_norm:
      self.norm = nn.LayerNorm(outputs_dim)
    else:
      self.norm = None

  def forward(self, x, state, masks):
    print('size ', x.size(0))
    if x.size(0) == state.size(0):
      # x is a [N, -1] tensor, state is a [N, 1, -1] tensor
      x, state = self.rnn(x.unsqueeze(0),
                (state * masks.repeat(1, self._rnn_layers).unsqueeze(-1)).transpose(0, 1).contiguous())
      x = x.squeeze(0)
    else:
      # Let's figure out which steps in the sequence have a zero for any agent
      # We will always assume t=0 has a zero in it as that makes the logic cleaner
      has_zeros = ((masks[1:] == 0.0)
             .any(dim=-1)
             .nonzero()
             .squeeze()
             .cpu())

      # +1 to correct the masks[1:]
      if has_zeros.dim() == 0:
        # Deal with scalar
        has_zeros = [has_zeros.item() + 1]
      else:
        has_zeros = (has_zeros + 1).numpy().tolist()

      # add t=0 and t=T to the list
      has_zeros = [0] + has_zeros + [x.size(0)]

      outputs = []
      for i in range(len(has_zeros) - 1):
        # We can now process steps that don't have any zeros in masks together!
        # This is much faster
        start_idx = has_zeros[i]
        end_idx = has_zeros[i + 1]
        temp = (state * masks[start_idx].view(1, -1, 1).repeat(self._rnn_layers, 1, 1)).contiguous()
        rnn_scores, state = self.rnn(x[start_idx:end_idx], temp)
        outputs.append(rnn_scores)

      # assert len(outputs) == T
      # x is a (T, N, -1) tensor
      x = torch.cat(outputs, dim=0)

    if self.norm:
      x = self.norm(x)
    return x, state
