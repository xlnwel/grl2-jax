import os
import warnings
from typing import Any, List
import ray

from core.utils import configure_gpu, set_seed


class RayBase:
    def __init__(self, id=None, seed=None):
        os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"
        warnings.filterwarnings("ignore")
        configure_gpu()
        if seed is not None:
            if id is not None:
                seed += id * 1000
            set_seed(seed)

    def register_handler(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @classmethod
    def as_remote(cls, **kwargs):
        if kwargs:
            return ray.remote(**kwargs)(cls)
        return ray.remote(cls)

    def exit(self):
        ray.actor.exit_actor()


class ManagerBase:
    def _remote_call(self, remotes: List, func: str, wait: bool=False):
        ids = [getattr(r, func).remote() for r in remotes]
        return self._wait(ids, wait=wait)

    def _remote_call_with_value(self, remotes: List, func: str, x: Any, wait: bool=False):
        oid = ray.put(x)
        ids = [getattr(r, func).remote(oid) for r in remotes]
        return self._wait(ids, wait=wait)

    def _remote_call_with_args(self, remotes: List, func: str, *args, wait: bool=False, **kwargs):
        args = [ray.put(x) for x in args]
        kwargs = {k: ray.put(v) for k, v in kwargs.items()}
        ids = [getattr(r, func).remote(*args, **kwargs) for r in remotes]
        return self._wait(ids, wait)

    def _remote_call_with_list(self, remotes: List, func: str, xs: List[Any], wait: bool=False):
        assert len(remotes) == len(xs), (len(remotes), len(xs))
        ids = [getattr(r, func).remote(x) for r, x in zip(remotes, xs)]
        return self._wait(ids, wait)

    def _wait(self, ids, wait=False):
        return ray.get(ids) if wait else ids
