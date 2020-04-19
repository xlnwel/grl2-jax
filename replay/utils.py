import numpy as np

from utility.utils import infer_dtype


def init_buffer(buffer, pre_dims, has_steps=False, precision=32, **kwargs):
    buffer.clear()
    if isinstance(pre_dims, int):
        pre_dims = [pre_dims]
    assert isinstance(pre_dims, (list, tuple))
    # v in buffer should have the same shape as v in kwargs except those specified by pre_dims
    info = infer_info(precision=precision, **kwargs)
    buffer.update(
        {k: np.zeros([*pre_dims, *v_shape], v_dtype) 
            if v_dtype else [None for _ in range(pre_dims[0])]
            for k, (v_shape, v_dtype) in info.items()})
    # we define an additional item, steps, that specifies steps in multi-step learning
    if has_steps:
        buffer['steps'] = np.ones(pre_dims, np.uint8)

def add_buffer(buffer, idx, n_steps, gamma, cycle=False, **kwargs):
    for k in buffer.keys():
        if k != 'steps':
            buffer[k][idx] = kwargs[k]

    # Update previous experience if multi-step is required
    for i in range(1, n_steps):
        k = idx - i
        if (k < 0 and not cycle) or buffer['discount'][k] == 0:
            break
        buffer['reward'][k] += gamma**i * kwargs['reward']
        buffer['discount'][k] = kwargs['discount']
        if 'steps' in buffer:
            buffer['steps'][k] += 1
        if 'nth_obs' in buffer:
            buffer['nth_obs'][k] = kwargs['nth_obs']

def copy_buffer(dest_buffer, dest_start, dest_end, orig_buffer, orig_start, orig_end, dest_keys=True):
    assert dest_end - dest_start == orig_end - orig_start, (
            f'Inconsistent lengths of dest_buffer(dest_end - dest_start)'
            f'and orig_buffer({orig_end - orig_start}).')
    if dest_end - dest_start == 0:
        return
    
    for key in (dest_buffer if dest_keys else orig_buffer).keys():
        dest_buffer[key][dest_start: dest_end] = orig_buffer[key][orig_start: orig_end]

def infer_info(precision, **kwargs):
    """ infer shape/type from kwargs so that we can use them for buffer initialization """
    info = {}
    pre_dims_len = 0 if isinstance(kwargs['reward'], (int, float)) \
        else len(kwargs['reward'].shape)
    for k, v in kwargs.items():
        v = np.array(v)
        dtype = infer_dtype(v.dtype, precision)
        if 'obs' in k and np.issubdtype(v.dtype, np.uint8):
            info[k] = ((), None)
        else:
            info[k] = (v.shape[pre_dims_len:], dtype)
        
    return info

def print_buffer(buffer):
    print('Buffer info')
    for k, v in buffer.items():
        shape = v.shape if isinstance(v, np.ndarray) else (len(v), np.array(v[0]).shape)
        dtype = v.dtype if isinstance(v, np.ndarray) else list
        print(f'\t{k}: shape({shape}), type({dtype})')
