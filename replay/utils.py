import numpy as np

def init_buffer(buffer, pre_dims, **kwargs):
    buffer.clear()
    if isinstance(pre_dims, int):
        pre_dims = [pre_dims]
    assert isinstance(pre_dims, (list, tuple))
    # v in buffer should have the same shape as v in kwargs except those specified by pre_dims
    info = infer_info(**kwargs)
    buffer.update(
        dict([(k, np.zeros([*pre_dims, *v_shape], v_dtype)) 
            for k, (v_shape, v_dtype) in info.items()]))
    # we define an additional item, steps, that specifies steps in multi-step learning
    # we define it even for 1-step learning to avoid code complication
    buffer['steps'] = np.ones([*pre_dims], np.uint8)
    print('Buffer info')
    for k, v in buffer.items():
        print(f'{k}: shape({v.shape}), type({v.dtype})')

def add_buffer(buffer, idx, n_steps, gamma, cycle=False, **kwargs):
    for k in buffer.keys():
        if k != 'steps':
            buffer[k][idx] = kwargs[k]

    # Update previous experience if multi-step is required
    for i in range(1, n_steps):
        k = idx - i
        if (k < 0 and not cycle) or buffer['done'][k]:
            break
        buffer['reward'][k] += gamma**i * kwargs['reward']
        buffer['done'][k] = kwargs['done']
        buffer['steps'][k] += 1
        if 'next_state' in buffer:
            buffer['next_state'][k] = kwargs['next_state']

def copy_buffer(dest_buffer, dest_start, dest_end, orig_buffer, orig_start, orig_end, dest_keys=True):
    assert dest_end - dest_start == orig_end - orig_start, (
            f'Inconsistent lengths of dest_buffer(dest_end - dest_start)'
            f'and orig_buffer({orig_end - orig_start}).')
    if dest_end - dest_start == 0:
        return
    
    for key in (dest_buffer if dest_keys else orig_buffer).keys():
        dest_buffer[key][dest_start: dest_end] = orig_buffer[key][orig_start: orig_end]

def infer_info(**kwargs):
    """ infer shape/type from kwargs so that we can use them for buffer initialization """
    info = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            info[k] = (v.shape, v.dtype)
        elif isinstance(v, (int, float, bool, np.float32)):
            # else assume v is of built-in type
            info[k] = ((), type(v))
        else:
            raise TypeError(f'v of type({v.dtype}) is not supported here')
    
    return info