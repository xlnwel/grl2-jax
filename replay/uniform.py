import collections
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.typing import AttrDict


class UniformReplay(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        super().__init__(config, env_stats, model, aid)

        self.max_size = int(self.config.max_size)
        self.min_size = int(self.config.min_size)
        self.batch_size = self.config.batch_size
        self.n_steps = self.config.n_steps

        self._memory = collections.defaultdict(
            collections.deque(maxlen=config.max_size))
        self._mem_idx = 0

    def reset(self):
        pass

    def collect(self, reset, obs, next_obs, **kwargs):
        for k, v in obs.items():
            if k not in kwargs:
                kwargs[k] = v
        for k, v in next_obs.items():
            kwargs[f'next_{k}'] = v
        self.add(**kwargs, reset=reset)

    def add(self, **data):
        for k, v in data.items():
            if k.startswith('pnext'):   # "next *" for the previous item
                if self._mem_idx == 0:
                    continue
                k = k[1:]   # k.replace('pnext', 'next')
                self._memory[k][-1] = v
            else:
                self._memory[k].append(v)
        self._mem_idx += 1

    def sample(self, batch_size=None):
        assert self.ready_to_sample(), (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self.min_size})')

        samples = self._sample(batch_size)

        return samples

    """ Implementation """
    def _sample(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        size = np.minimum(self._mem_idx, self.max_size-1)   # we omit the last item since it may miss some data
        idxes = np.random.randint(size, size=batch_size)
        # the following code avoids repetitive sampling, 
        # but it takes significant more time to run(around 1000x).
        # idxes = np.random.choice(size, size=batch_size, replace=False)
        
        samples = self._get_samples(idxes)

        return samples

    def _get_samples(self, idxes):
        results = {}
        for k, v in self._memory.items():
            results[k] = np.array([v[i] for i in idxes])

        return results
