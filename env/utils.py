from env.typing import EnvOutput
from utility.utils import convert_batch_with_func


def batch_env_output(out):
    return EnvOutput(*[convert_batch_with_func(o) for o in zip(*out)])


def compute_aid2pids(pid2aid):
    aid2pids = []
    for pid, aid in enumerate(pid2aid):
        if aid > len(aid2pids):
            raise ValueError(f'pid2aid({pid2aid}) is not sorted in order')
        if aid == len(aid2pids):
            aid2pids.append((pid, ))
        else:
            aid2pids[aid] += (pid,)
    return aid2pids
