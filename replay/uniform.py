import collections
from typing import List
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.typing import AttrDict, tree_slice, subdict
from replay.local import NStepBuffer
from tools.utils import batch_dicts
from replay import replay_registry


@replay_registry.register('uniform')
class UniformReplay(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        super().__init__(config, env_stats, model, aid)

        self.n_envs = self.config.n_runners * self.config.n_envs

        self.max_size = int(self.config.max_size)
        self.min_size = int(self.config.min_size)
        self.batch_size = self.config.batch_size
        self.n_recency = self.config.get('n_recency', self.min_size)
        self.n_steps = self.config.n_steps

        self._memory = collections.deque(maxlen=self.max_size)
        self._ret_latest = config.ret_latest_data
        if self._ret_latest:
            self._latest_memory = None

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
    
    def collect_and_pop(self, idxes=None, **data):
        data = self._prepare_data(**data)
        return self.add_and_pop(idxes=idxes, **data)

    def add(self, idxes=None, **data):
        if self.n_envs > 1:
            if idxes is None:
                idxes = range(self.n_envs)
            for i in idxes:
                d = tree_slice(data, i)
                if i >= len(self._tmp_bufs):
                    self._tmp_bufs.append(NStepBuffer(
                        self.config, self.env_stats, self.model, self.aid, 0))
                traj = self._tmp_bufs[i].add(**d)
                if traj is not None:
                    self.merge(traj)
        else:
            traj = self._tmp_bufs[0].add(**data)
            if traj is not None:
                self.merge(traj)

    def add_and_pop(self, idxes=None, **data):
        popped_data = []
        if self.n_envs > 1:
            if idxes is None:
                idxes = range(self.n_envs)
            for i in idxes:
                d = tree_slice(data, i)
                if i >= len(self._tmp_bufs):
                    self._tmp_bufs.append(NStepBuffer(
                        self.config, self.env_stats, self.model, self.aid, 0))
                traj = self._tmp_bufs[i].add(**d)
                if traj is not None:
                    popped_data.extend(self.merge_and_pop(traj))
        else:
            traj = self._tmp_bufs[0].add(**data)
            if traj is not None:
                popped_data.extend(self.merge_and_pop(traj))

        return popped_data

    def merge(self, trajs):
        if isinstance(trajs, dict):
            trajs = [trajs]
        self._memory.extend(trajs)
        if self._ret_latest:
            self._latest_memory = trajs

    def merge_and_pop(self, trajs):
        if isinstance(trajs, dict):
            trajs = [trajs]
        popped_data = []
        for traj in trajs:
            if len(self._memory) == self._memory.maxlen:
                popped_data.append(self._memory.popleft())
            self._memory.append(traj)
        return popped_data

    def sample_from_recency(self, batch_size, sample_keys, n=None, **kwargs):
        batch_size = batch_size or self.batch_size
        n = max(batch_size, n or self.n_recency)
        if n <= len(self._memory):
            return None
        idxes = np.arange(len(self)-n, len(self))
        idxes = np.random.choice(idxes, size=batch_size, replace=False)

        samples = batch_dicts([subdict(self._memory[i], *sample_keys) for i in idxes])

        return samples
        
    def sample(self, batch_size=None):
        if self.ready_to_sample():
            samples = self._sample(batch_size)
        else:
            samples = None

        return samples
    
    def ergodic_sample(self, batch_size=None):
        if not self.ready_to_sample():
            return None
        batch_size = batch_size or self.batch_size
        idxes = np.arange(0, len(self))
        maxlen = (len(self) // batch_size) * batch_size
        idxes = idxes[-maxlen:]
        np.random.shuffle(idxes)
        i = 0
        for i in range(0, maxlen, batch_size):
            if self._ret_latest:
                yield self._get_samples(range(len(self._latest_memory)), self._latest_memory), self._get_samples(idxes[i:i+batch_size], self._memory)
            else:
                yield self._get_samples(idxes[i:i+batch_size], self._memory)
    
    def retrieve_all_data(self):
        data = self._memory
        self._memory = collections.deque(max_len=self.config.max_size)
        return data

    def clear_local_buffer(self, drop_data=False):
        for b in self._tmp_bufs:
            if drop_data:
                b.reset()
            else:
                traj = b.retrieve_all_data()
                self.merge(traj)

    def _sample(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        idxes = np.random.randint(len(self), size=batch_size)
        # the following code avoids repetitive sampling, 
        # but it takes significant more time to run(around 1000x).
        # idxes = np.random.choice(size, size=batch_size, replace=False)
        
        samples = self._get_samples(idxes, self._memory)
        if self._ret_latest:
            latest_samples = self._get_samples(range(len(self._latest_memory)), self._latest_memory)
            return latest_samples, samples

        return samples

    def _get_samples(self, idxes, memory):
        fn = lambda x: np.expand_dims(np.stack(x), 1)
        results = batch_dicts([memory[i] for i in idxes], func=fn)

        return results
