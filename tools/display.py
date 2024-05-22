from typing import Sequence, Dict
import numpy as np
import jax
import torch

from tools.log import do_logging


def print_dict(tree, prefix='', backtrack=3, **kwargs):
  if isinstance(tree, tuple) and hasattr(tree, '_asdict'):
    print_dict(tree._asdict(), prefix=prefix, backtrack=backtrack+1, **kwargs)
  elif isinstance(tree, Dict):
    for k, v in tree.items():
      new_prefix = prefix + '.' if prefix else prefix
      new_prefix += f'{k}'
      if isinstance(v, Dict):
        do_logging(new_prefix, backtrack=backtrack, **kwargs)
        print_dict(v, prefix='\t'+new_prefix, backtrack=backtrack+1, **kwargs)
      else:
        print_dict(v, prefix=new_prefix, backtrack=backtrack+1, **kwargs)
  elif isinstance(tree, (list, tuple)):
    prefix = prefix + '.' if prefix else prefix
    line = []
    print_line = False
    for i, v in enumerate(tree):
      if isinstance(v, dict):
        new_prefix = f'{prefix}.{i}'
        do_logging(new_prefix, backtrack=backtrack, **kwargs)
        print_dict(v, prefix='\t'+new_prefix, backtrack=backtrack+1, **kwargs)
      else:
        line.append(f'{v}')
        # do_logging(f'{prefix}: {v}', backtrack=backtrack+1, **kwargs)
        print_line = True
    if print_line:
      line = prefix + ' ' + ','.join(line)
      do_logging(line, backtrack=backtrack, **kwargs)
  else:
    do_logging(f'{prefix}: {tree}', backtrack=backtrack, **kwargs)


def print_array(v, prefix, backtrack, **kwargs):
  do_logging(
    f'{prefix}: {v.shape} {v.dtype} '
    f'norm({np.linalg.norm(v):0.4g}) ' \
    f'mean({v.mean():0.4g}) ' \
    f'std({v.std():0.4g}) ' \
    f'max({v.max():0.4g}) ' \
    f'min({v.min():0.4g})',
    backtrack=backtrack+1, **kwargs)


def print_dict_info(tree, prefix='', backtrack=3, **kwargs):
  if isinstance(tree, tuple) and hasattr(tree, '_asdict'):
    print_dict_info(tree._asdict(), prefix+'\t', backtrack=backtrack+1, **kwargs)
  elif isinstance(tree, (list, tuple)):
    for i, v in enumerate(tree):
      print_dict_info(v, f'{prefix} idx({i})', backtrack=backtrack+1, **kwargs)
  elif isinstance(tree, (np.ndarray, jax.Array, torch.Tensor)):
    print_array(tree, prefix, backtrack=backtrack, **kwargs)
  elif isinstance(tree, Dict):
    for k, v in tree.items():
      if isinstance(v, Dict):
        do_logging(f'{prefix} {k}', backtrack=backtrack, **kwargs)
        print_dict_info(v, prefix+'\t', backtrack=backtrack+1, **kwargs)
      elif isinstance(v, tuple) and hasattr(v, '_asdict'):
        # namedtuple is assumed
        do_logging(f'{prefix} {k}', backtrack=backtrack, **kwargs)
        print_dict_info(v._asdict(), prefix+'\t', backtrack=backtrack+1, **kwargs)
      elif isinstance(v, (Sequence)):
        do_logging(f'{prefix} {k} length: {len(v)}', backtrack=backtrack, **kwargs)
        print_dict_info(v, f'{prefix} {k}', backtrack=backtrack+1, **kwargs)
      elif isinstance(v, (np.ndarray, jax.Array, torch.Tensor)):
        print_array(v, f'{prefix} {k}', backtrack=backtrack, **kwargs)
      else:
        do_logging(f'{prefix} {k}: {v} {type(v)}', backtrack=backtrack, **kwargs)
  else:
    do_logging(f'{prefix}: {tree}', backtrack=backtrack, **kwargs)


def summarize_arrays(tree):
  n = 0
  if isinstance(tree, (Sequence)):
    n += sum([summarize_arrays(t) for t in tree])
  elif isinstance(tree, Dict):
    n += sum([summarize_arrays(v) for v in tree.values()])
  elif hasattr(tree, 'size'):
    n += tree.size
  return n


def int2str(step):
  if step < 1000:
    return f'{step}'
  elif step < 1000000:
    return f'{step/1000:.3g}k'
  else:
    return f'{step/1000000:.3g}m'
