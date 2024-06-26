import numpy as np
from jax.tree_util import tree_map
import torch


def to_tensor(data, tpdv):
  data = tree_map(
    lambda x: torch.from_numpy(x).to(**tpdv) 
    if isinstance(x, np.ndarray) else x.to(**tpdv), data
  )
  return data

def to_numpy(data):
  data = tree_map(
    lambda x: x.detach().cpu().numpy()
    if isinstance(x, torch.Tensor) else x, data)
  return data

def static_scan(f, inputs, start):
  """
    One function which can help conduct trajectory roll-out or performing rnn
    It can be seem as a general version or an extension of hk.static_unroll
  """
  last = start
  if isinstance(start, tuple) and hasattr(start, "_asdict"):
    _keys = start._asdict().keys()
    outputs = {
      _key: [start._asdict()[_key]] for _key in _keys
    }
  elif isinstance(start, tuple):
    assert isinstance(start[0], tuple), start[0].__class__
    _keys = start[0]._asdict().keys()
    outputs = [{
      _key: [start[i]._asdict()[_key]] for _key in _keys
    } for i in range(len(start))]
  else:
    assert 0, start.__class__

  assert isinstance(inputs, tuple) or isinstance(inputs, jnp.ndarray), inputs.__class__
  indices = range(inputs.shape[0]) if isinstance(inputs, jnp.ndarray) else range(inputs[-1].shape[0])
  for index in indices:
    inp = inputs[index] if isinstance(inputs, jnp.ndarray) else tuple([inputs_item[index] if inputs_item is not None else None for inputs_item in inputs])
    last = f(last, inp)
    if isinstance(last, tuple) and hasattr(last, "_asdict"):
      for _key in outputs:
        outputs[_key].append(eval(f"last.{_key}"))
    else:
      for i in range(len(last)):
        for _key in outputs[i]:
          outputs[i][_key].append(eval(f"last[{i}].{_key}"))
  if isinstance(outputs, dict):
    for _key in outputs:
      outputs[_key] = torch.stack(outputs[_key], 0)
  elif isinstance(outputs, list):
    for i in range(len(outputs)):
      for _key in outputs[i]:
        outputs[i][_key] = torch.stack(outputs[i][_key], 0)
  else:
    assert 0, outputs.__class__
  return outputs

def split_data(x, next_x=None, axis=1):
  if isinstance(x, (list, tuple)):
    if next_x is None:
      next_x = [None for _ in x]
    x, next_x = list(zip(*[split_data(xx, next_xx, axis=axis) 
      for xx, next_xx in zip(x, next_x)]))
    return x, next_x
  if x is None:
    return None, None
  if next_x is None:
    slices = [slice(None) for _ in x.shape]
    slices[axis] = slice(1, None)
    next_x = x[tuple(slices)]
    n = x.shape[axis]
    slices[axis] = slice(0, n-1)
    x = x[tuple(slices)]

  return x, next_x

def time_major(*args, axis):
  dims = list(range(args[0].ndim))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if len(args) == 1:
    args = args[0]
  if axis != 0:
    args = tree_map(lambda x: torch.permute(x, dims), args)
  return dims, args

def undo_time_major(*args, dims, axis):
  if len(args) == 1:
    args = args[0]
  if axis != 0:
    args = tree_map(lambda x: torch.permute(x, dims), args)
  return args

def compute_norms(tree):
  tree = tree_map(torch.linalg.norm, tree)
  return tree
