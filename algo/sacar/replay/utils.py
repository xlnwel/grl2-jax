import numpy as np

def init_buffer(buffer, *keys, capacity=0, state_shape=None):
    buffer.clear()
    buffer.update(dict([(key, [None] * capacity) for key in keys]))
    if 'next_state' not in buffer and state_shape is not None:
        default_state = np.zeros(state_shape)
        buffer['state'] = [default_state] * capacity

def add_buffer(buffer, idx, n_steps, gamma, cycle=False, **kwargs):
    for k in buffer.keys():
        if k == 'steps':
            buffer[k][idx] = 1#kwargs['n_ar']
        else:
            buffer[k][idx] = kwargs[k]

    # Update previous experience if multi-step is required
    for i in range(1, n_steps):
        k = idx - i
        if (k < 0 and not cycle) or buffer['done'][k]:
            break
        buffer['reward'][k] += gamma**i * kwargs['reward']
        buffer['done'][k] = kwargs['done']
        buffer['steps'][k] += 1#kwargs['n_ar']
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
