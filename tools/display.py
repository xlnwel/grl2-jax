from datetime import datetime
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

def print_dict_info(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            do_logging(f'{prefix} {k}')
            print_dict_info(v, prefix+'\t')
        elif isinstance(v, tuple) and hasattr(v, '_asdict'):
            # namedtuple is assumed
            do_logging(f'{prefix} {k}')
            print_dict_info(v._asdict(), prefix+'\t')
        elif isinstance(v, (tuple, list)):
            do_logging(f'{prefix} {k}: {len(v)}')
        elif isinstance(v, (np.ndarray, jnp.DeviceArray, jax.ShapedArray)):
            do_logging(f'{prefix} {k}: {v.shape} {v.dtype}')
        else:
            do_logging(f'{prefix} {k}: {v}')
