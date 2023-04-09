from typing import Sequence, Dict
import numpy as np
import jax
import jax.numpy as jnp

from core.log import do_logging


def print_dict(tree, prefix='', level='pwt', backtrack=3):
    if isinstance(tree, tuple) and hasattr(tree, '_asdict'):
        print_dict(tree._asdict(), prefix+'\t', level=level, backtrack=backtrack+1)
    elif isinstance(tree, Dict):
        for k, v in tree.items():
            if isinstance(v, Dict):
                do_logging(f'{prefix} {k}')
                print_dict(v, prefix=f'{prefix} {k}\t', level=level, backtrack=backtrack+1)
            else:
                print_dict(v, prefix=f'{prefix} {k}\t', level=level, backtrack=backtrack+1)
    else:
        do_logging(f'{prefix}: {tree}', level=level, backtrack=backtrack)


def print_array(v, prefix, level, backtrack):
    do_logging(f'{prefix}: {v.shape} {v.dtype} '
        f'norm({np.linalg.norm(v):0.4g}) mean({v.mean():0.4g}) std({v.std():0.4g}) max({v.max():0.4g}) min({v.min():0.4g}) ',
        level=level, backtrack=backtrack+1)

def print_dict_info(tree, prefix='', level='pwt', backtrack=3):
    if isinstance(tree, tuple) and hasattr(tree, '_asdict'):
        print_dict_info(tree._asdict(), prefix+'\t', level=level, backtrack=backtrack+1)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            print_dict_info(v, f'{prefix} idx({i})', level=level, backtrack=backtrack+1)
    elif isinstance(tree, (np.ndarray, jnp.DeviceArray, jax.ShapedArray)):
        print_array(tree, prefix, level, backtrack=backtrack)
    elif isinstance(tree, Dict):
        for k, v in tree.items():
            if isinstance(v, Dict):
                do_logging(f'{prefix} {k}', level=level, backtrack=backtrack)
                print_dict_info(v, prefix+'\t', level=level, backtrack=backtrack+1)
            elif isinstance(v, tuple) and hasattr(v, '_asdict'):
                # namedtuple is assumed
                do_logging(f'{prefix} {k}', level=level, backtrack=backtrack)
                print_dict_info(v._asdict(), prefix+'\t', level=level, backtrack=backtrack+1)
            elif isinstance(v, (Sequence)):
                do_logging(f'{prefix} {k}: {len(v)}', level=level, backtrack=backtrack)
                print_dict_info(v, f'{prefix} {k}', level, backtrack=backtrack+1)
            elif isinstance(v, (np.ndarray, jnp.DeviceArray, jax.ShapedArray)):
                print_array(v, f'{prefix} {k}', level, backtrack=backtrack)
            else:
                do_logging(f'{prefix} {k}: {v} {type(v)}', level=level, backtrack=backtrack)
    else:
        do_logging(f'{prefix}: {tree}', level=level, backtrack=backtrack)

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
