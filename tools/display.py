from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp

from core.log import do_logging


def print_dict(d, prefix='', level='pwt', backtrack=3):
    if isinstance(d, Sequence):
        for v in d:
            print_dict(v, prefix, level=level, backtrack=backtrack+1)
    for k, v in d.items():
        if isinstance(v, dict):
            do_logging(f'{prefix} {k}')
            print_dict(v, prefix+'\t')
        elif isinstance(v, tuple) and hasattr(v, '_asdict'):
            # namedtuple is assumed
            do_logging(f'{prefix} {k}')
            print_dict(v._asdict(), prefix+'\t')
        else:
            do_logging(f'{prefix} {k}: {v}')

def _print_array(v, prefix, level, backtrack):
    do_logging(f'{prefix}: {v.shape} {v.dtype} '
        f'norm({np.linalg.norm(v):0.4g}) mean({v.mean():0.4g}) max({v.max():0.4g}) min({v.min():0.4g}) ',
        level=level, backtrack=backtrack+1)

def print_dict_info(d, prefix='', level='pwt', backtrack=3):
    if isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            print_dict_info(v, f'{prefix} {i}', level=level, backtrack=backtrack+1)
    elif isinstance(d, (np.ndarray, jnp.DeviceArray, jax.ShapedArray)):
        _print_array(d, prefix, level, backtrack=backtrack)
    else:
        for k, v in d.items():
            if isinstance(v, dict):
                do_logging(f'{prefix} {k}', level=level, backtrack=backtrack)
                print_dict_info(v, prefix+'\t', level=level, backtrack=backtrack+1)
            elif isinstance(v, tuple) and hasattr(v, '_asdict'):
                # namedtuple is assumed
                do_logging(f'{prefix} {k}', level=level, backtrack=backtrack)
                print_dict_info(v._asdict(), prefix+'\t', level=level, backtrack=backtrack+1)
            elif isinstance(v, (list, tuple)):
                do_logging(f'{prefix} {k}: {len(v)}', level=level, backtrack=backtrack)
                print_dict_info(v, f'{prefix} {k}', level, backtrack=backtrack+1)
            elif isinstance(v, (np.ndarray, jnp.DeviceArray, jax.ShapedArray)):
                _print_array(v, f'{prefix} {k}', level, backtrack=backtrack)
            else:
                do_logging(f'{prefix} {k}: {v} {type(v)}', level=level, backtrack=backtrack+1)
