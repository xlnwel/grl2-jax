import collections
from typing import List
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.typing import AttrDict, tree_slice
from replay.local import NStepBuffer
from tools.utils import batch_dicts
from replay import buffer_registry


@buffer_registry.register('uniform')
class UniformReplay(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        super().__init__(config, env_stats, model, aid)

        self.n_runners = self.config.n_runners
        self.n_envs = self.n_runners * self.config.n_envs

        self.max_size = int(self.config.max_size)
        self.min_size = int(self.config.min_size)
        self.batch_size = self.config.batch_size
        self.n_steps = self.config.n_steps

        self._memory = collections.deque(maxlen=config.max_size)

        self._tmp_bufs: List[NStepBuffer] = [
            NStepBuffer(config, env_stats, model, aid, 0) 
            for _ in range(self.n_envs)
        ]

    def __len__(self):
        return len(self._memory)

    def ready_to_sample(self):
        return len(self._memory) >= self.min_size

    def reset(self):
        pass

    def collect(self, reset, obs, next_obs, **kwargs):
        for k, v in obs.items():
            if k not in kwargs:
                kwargs[k] = v
        for k, v in next_obs.items():
            kwargs[f'next_{k}'] = v
        self.add(**kwargs, reset=reset)

    def add(self, idxes=None, **data):
        if self.n_envs > 1:
            if idxes is None:
                idxes = range(self.n_envs)
            for i in idxes:
                d = tree_slice(data, i)
                traj = self._tmp_bufs[i].add(**d)
                if traj is not None:
                    self.merge(traj)
        else:
            traj = self._tmp_bufs[0].add(**data)
            if traj is not None:
                self.merge(traj)

    def merge(self, trajs):
        if isinstance(trajs, dict):
            trajs = [trajs]
        self._memory.extend(trajs)

    def sample(self, batch_size=None):
        if self.ready_to_sample():
            samples = self._sample(batch_size)
        else:
            samples = None

        return samples

    """ Implementation """
    def _sample(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        idxes = np.random.randint(len(self), size=batch_size)
        # the following code avoids repetitive sampling, 
        # but it takes significant more time to run(around 1000x).
        # idxes = np.random.choice(size, size=batch_size, replace=False)
        
        samples = self._get_samples(idxes)

        return samples

    def _get_samples(self, idxes):
        fn = lambda x: np.expand_dims(np.stack(x), 1)
        results = batch_dicts([self._memory[i] for i in idxes], func=fn)

        return results
