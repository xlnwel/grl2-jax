from algo.zero.elements.utils import *

def get_hx(idx, event):
    if idx is None:
        hx = event
    elif event is None:
        hx = idx
    else:
        hx = tf.concat([idx, event], -1)
    return hx
