import logging
import torch


logger = logging.getLogger(__name__)


def select_optimizer(name):
  # add custom optimizers here
  if isinstance(name, str):
    return getattr(torch.optim, name)
  return name


def build_optimizer(
  *,
  params=None, 
  opt_name='Adam', 
  lr, 
  weight_decay: float=0., 
  **opt_kwargs, 
):
  opt = select_optimizer(opt_name)(params, lr=lr, weight_decay=weight_decay, **opt_kwargs)
  return opt


def optimize(opt, loss, params, clip_norm):
  opt.zero_grad()
  loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(params, clip_norm)
  opt.step()
  return norm


if __name__ == '__main__':
  import numpy as np
  import torch.nn as nn
  eta = np.array(1, dtype='float32')
  x = np.random.uniform(size=(2, 3))
  x = torch.tensor(x, dtype=torch.float32)
  l = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
  y = l(x)
  opt = build_optimizer(params=l.parameters(), opt_name='Adam', lr=1e-4)
  print(dir(opt))
  print(opt.param_groups)
