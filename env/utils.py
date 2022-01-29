from env.typing import EnvOutput
from utility.utils import convert_batch_with_func


def batch_env_output(out):
    return EnvOutput(*[convert_batch_with_func(o) for o in zip(*out)])


def compute_aid2uids(uid2aid):
    """ Compute aid2uids from uid2aid """
    aid2uids = []
    for uid, aid in enumerate(uid2aid):
        if aid > len(aid2uids):
            raise ValueError(f'uid2aid({uid2aid}) is not sorted in order')
        if aid == len(aid2uids):
            aid2uids.append((uid, ))
        else:
            aid2uids[aid] += (uid,)

    return aid2uids
