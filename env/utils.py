import numpy as np

from env.typing import EnvOutput
from utility.utils import convert_batch_with_func


def batch_env_output(out, func=np.stack):
    return EnvOutput(*[convert_batch_with_func(o, func=func) for o in zip(*out)])


def batch_ma_env_output(out, func=np.stack):
    obs = [convert_batch_with_func(o, func=func) for o in zip(*out[0])]
    reward = [convert_batch_with_func(r, func=func) for r in zip(*out[1])]
    discount = [convert_batch_with_func(d, func=func) for d in zip(*out[2])]
    reset = [convert_batch_with_func(r, func=func) for r in zip(*out[3])]
    
    return EnvOutput(obs, reward, discount, reset)


def compute_aid2uids(uid2aid):
    """ Compute aid2uids from uid2aid """
    aid2uids = []
    for uid, aid in enumerate(uid2aid):
        # if aid > len(aid2uids):
        #     raise ValueError(f'uid2aid({uid2aid}) is not sorted in order')
        if aid == len(aid2uids):
            aid2uids.append((uid, ))
        else:
            aid2uids[aid] += (uid,)
    aid2uids = [np.array(uids, np.int32) for uids in aid2uids]

    return aid2uids

def compute_relative_position(center, x):
    return x - center

def compute_angle_cos_sin(center, x):
    diff = x - center
    dist = np.linalg.norm(diff)
    ans = diff / dist
    return ans
