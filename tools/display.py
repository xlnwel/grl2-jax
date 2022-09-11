import numpy as np
import jax
import jax.numpy as jnp

from core.log import do_logging


def print_dict(d, prefix=''):
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

def print_dict_info(d, prefix='', level='pwt', backtrack=3):
    if isinstance(d, (list, tuple)):
        for v in d:
            print_dict_info(d, prefix, level=level, backtrack=backtrack+1)
    for k, v in d.items():
        if isinstance(v, dict):
            do_logging(f'{prefix} {k}', level=level, backtrack=backtrack)
            print_dict_info(v, prefix+'\t', level=level, backtrack=backtrack+1)
        elif isinstance(v, tuple) and hasattr(v, '_asdict'):
            # namedtuple is assumed
            do_logging(f'{prefix} {k}', level=level, backtrack=backtrack)
            print_dict_info(v._asdict(), prefix+'\t', level=level, backtrack=backtrack+1)
        elif isinstance(v, (tuple, list)):
            do_logging(f'{prefix} {k}: {len(v)}', level=level, backtrack=backtrack)
        elif isinstance(v, (np.ndarray, jnp.DeviceArray, jax.ShapedArray)):
            do_logging(f'{prefix} {k}: {v.shape} {v.dtype} '
                f'norm({np.linalg.norm(v):0.4g}) max({v.max():0.4g}) min({v.min():0.4g}) ',
                level=level, backtrack=backtrack)
        else:
            do_logging(f'{prefix} {k}: {v}', level=level, backtrack=backtrack+1)
